#!/usr/bin/env python3
"""
Concept importance analysis for MIMIC-CXR opportunistic phenotype linear probes.

For each of the 27 phenotype labels, computes which radiological concepts are
most positively / negatively aligned with the trained linear weight vector.

  importance(concept_i, label_j) = cos_sim(W_j, E_i)

where W_j is the learned weight vector for label j, and E_i is the LLM embedding
of concept i.

Supports aggregation across multiple random seeds (mean alignment).

Usage:
  # After running exp_linear_mimic.py:
  python concept_importance_mimic.py --results_dir results/linear_mimic_openai_3large
  python concept_importance_mimic.py --results_dir results/linear_mimic_sfr_mistral --top_k 50
  python concept_importance_mimic.py --results_dir results/linear_mimic_openai_3large --seed 42
"""
import os
import sys
import json
import glob
import pickle
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

# Import model class from the linear probing script
sys.path.insert(0, os.path.dirname(__file__))
from exp_linear_mimic import (
    LogisticRegressionModel, EMBEDDING_MODELS, EMBEDDINGS_DIR,
    CONCEPTS_CSV, SKIP_EVAL_LABELS, CORE_PHENOTYPES,
)

# Human-readable label names
LABEL_TO_PROMPT = {
    'heart_failure': 'heart failure',
    'hfref': 'heart failure with reduced ejection fraction',
    'hfpef': 'heart failure with preserved ejection fraction',
    'atrial_fibrillation': 'atrial fibrillation',
    'mitral_valve': 'mitral valve disease',
    'aortic_valve': 'aortic valve disease',
    'tricuspid_valve': 'tricuspid valve disease',
    'pulmonary_valve': 'pulmonary valve disease',
    'cad': 'coronary artery disease',
    'myocardial_infarction': 'myocardial infarction',
    'pulmonary_htn': 'pulmonary hypertension',
    'stroke': 'stroke',
    'copd': 'chronic obstructive pulmonary disease',
    'asthma': 'asthma',
    'ild': 'interstitial lung disease',
    'lung_cancer': 'lung cancer',
    'tuberculosis': 'tuberculosis',
    't2dm': 'type 2 diabetes',
    'obesity': 'obesity',
    'dyslipidemia': 'dyslipidemia',
    'osteoporosis': 'osteoporosis',
    'spinal_stenosis': 'spinal stenosis',
    'hypertension': 'hypertension',
    'ckd': 'chronic kidney disease',
    'pvd': 'peripheral vascular disease',
    'liver_cirrhosis': 'liver cirrhosis',
    'anemia': 'anemia',
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_concepts():
    """Load concept texts and indices."""
    df = pd.read_csv(CONCEPTS_CSV)
    concepts = df['concept'].tolist()
    indices = df['concept_idx'].tolist()
    print(f"Loaded {len(concepts)} concepts")
    return concepts, indices


def load_concept_embeddings(model_key: str, concept_indices: List[int],
                            concept_texts: List[str] = None):
    """Load precomputed LLM embeddings. Supports both int-keyed and string-keyed pickles."""
    info = EMBEDDING_MODELS[model_key]
    pickle_path = os.path.join(EMBEDDINGS_DIR,
                               f"cxr_embeddings_{info['pickle_suffix']}.pickle")
    print(f"Loading concept embeddings: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    first_key = next(iter(data))
    uses_string_keys = isinstance(first_key, str)
    if uses_string_keys:
        print(f"  Detected string-keyed pickle (e.g., {repr(first_key)[:60]})")
        data_lower = {k.lower(): v for k, v in data.items()}
    else:
        print(f"  Detected integer-keyed pickle (e.g., {first_key})")
        data_lower = None

    dim = info['dim']
    embeddings = np.zeros((len(concept_indices), dim), dtype=np.float32)
    missing = 0
    for pos, idx in enumerate(concept_indices):
        emb = None
        if not uses_string_keys:
            emb = data.get(idx)
        elif concept_texts is not None:
            text = concept_texts[pos]
            emb = data.get(text)
            if emb is None:
                emb = data_lower.get(text.lower())

        if emb is not None:
            if isinstance(emb, np.ndarray):
                embeddings[pos] = emb.astype(np.float32)
            else:
                embeddings[pos] = np.array(emb, dtype=np.float32)
        else:
            missing += 1

    if missing:
        print(f"  Warning: {missing}/{len(concept_indices)} concepts missing")
    else:
        print(f"  All {len(concept_indices)} concepts matched")

    return embeddings


def detect_embedding_model(results_dir: str) -> str:
    """Detect which embedding model was used from the results directory name."""
    dirname = os.path.basename(results_dir)
    # Directory format: linear_mimic_{model_key}
    for key in EMBEDDING_MODELS:
        if key in dirname:
            return key
    raise ValueError(f"Cannot detect embedding model from directory: {dirname}. "
                     f"Use --embedding_model to specify.")


def load_trained_models(results_dir: str,
                        seed: Optional[int] = None) -> List[dict]:
    """Load trained linear probe model checkpoints.

    If seed is specified, loads only that seed. Otherwise loads all.
    """
    models_dir = os.path.join(results_dir, 'models')
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    if seed is not None:
        path = os.path.join(models_dir, f"seed_{seed}_model.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        return [ckpt]

    # Load all seeds
    pattern = os.path.join(models_dir, "seed_*_model.pth")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No model checkpoints found in {models_dir}")

    checkpoints = []
    for p in paths:
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        checkpoints.append(ckpt)

    print(f"Loaded {len(checkpoints)} model checkpoints "
          f"(seeds: {[c['seed'] for c in checkpoints]})")
    return checkpoints


def load_summary(results_dir: str) -> dict:
    """Load the most recent summary JSON."""
    pattern = os.path.join(results_dir, "summary_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No summary file found in {results_dir}")
    with open(paths[-1]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Importance computation
# ---------------------------------------------------------------------------
def compute_concept_importance(weight_matrix: np.ndarray,
                                concept_embeddings: np.ndarray,
                                labels: List[str],
                                concepts: List[str],
                                concept_indices: List[int],
                                top_k: int = 100,
                                per_label_aucs: Optional[dict] = None):
    """
    Compute concept importance for each label via cosine similarity
    between the label's weight vector and each concept embedding.

    Args:
        weight_matrix: [N_labels, emb_dim] - linear layer weights
        concept_embeddings: [N_concepts, emb_dim] - LLM concept embeddings
        labels: list of label names
        concepts: list of concept text strings
        concept_indices: list of concept integer IDs
        top_k: number of top positive/negative concepts per label
        per_label_aucs: optional dict of {label: auc} for metadata

    Returns:
        dict of {label: {positive_concepts, negative_concepts, stats}}
    """
    # Vectorized cosine similarity: [N_labels, N_concepts]
    # Normalize weight vectors
    w_norms = np.linalg.norm(weight_matrix, axis=1, keepdims=True)
    w_norms = np.clip(w_norms, 1e-8, None)
    w_normed = weight_matrix / w_norms

    # Normalize concept embeddings
    e_norms = np.linalg.norm(concept_embeddings, axis=1, keepdims=True)
    e_norms = np.clip(e_norms, 1e-8, None)
    e_normed = concept_embeddings / e_norms

    # Cosine similarity matrix
    alignments = w_normed @ e_normed.T  # [N_labels, N_concepts]

    importance = {}
    for label_idx, label in enumerate(labels):
        if label in SKIP_EVAL_LABELS:
            continue

        align = alignments[label_idx]  # [N_concepts]

        # Separate positive and negative
        pos_mask = align > 0
        neg_mask = align < 0

        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        # Sort: positive descending, negative ascending (most negative first)
        pos_sorted = pos_indices[np.argsort(align[pos_indices])[::-1]]
        neg_sorted = neg_indices[np.argsort(align[neg_indices])]

        top_positive = [
            {
                'concept': concepts[idx],
                'concept_idx': concept_indices[idx],
                'alignment': float(align[idx]),
            }
            for idx in pos_sorted[:top_k]
        ]

        top_negative = [
            {
                'concept': concepts[idx],
                'concept_idx': concept_indices[idx],
                'alignment': float(align[idx]),
            }
            for idx in neg_sorted[:top_k]
        ]

        stats = {
            'label_readable': LABEL_TO_PROMPT.get(label, label),
            'is_core': label in CORE_PHENOTYPES,
            'total_positive': int(len(pos_indices)),
            'total_negative': int(len(neg_indices)),
            'max_positive': float(align[pos_sorted[0]]) if len(pos_sorted) > 0 else 0.0,
            'min_negative': float(align[neg_sorted[0]]) if len(neg_sorted) > 0 else 0.0,
        }
        if per_label_aucs and label in per_label_aucs:
            stats['auc'] = per_label_aucs[label]

        importance[label] = {
            'positive_concepts': top_positive,
            'negative_concepts': top_negative,
            'stats': stats,
        }

        # Print summary
        marker = " *" if label in CORE_PHENOTYPES else ""
        print(f"\n{label}{marker}: {len(pos_indices)} pos, "
              f"{len(neg_indices)} neg concepts")

        print(f"  Top 5 Positive:")
        for i, c in enumerate(top_positive[:5]):
            print(f"    {i+1}. +{c['alignment']:.4f} | {c['concept'][:70]}")

        print(f"  Top 5 Negative:")
        for i, c in enumerate(top_negative[:5]):
            print(f"    {i+1}. {c['alignment']:.4f} | {c['concept'][:70]}")

    return importance


def aggregate_weights(checkpoints: List[dict]) -> np.ndarray:
    """Average weight matrices across seeds."""
    weights = []
    for ckpt in checkpoints:
        model = LogisticRegressionModel(
            ckpt['model_config']['input_dim'],
            ckpt['model_config']['output_dim'],
        )
        model.load_state_dict(ckpt['state_dict'])
        w = model.linear.weight.detach().cpu().numpy()
        weights.append(w)

    mean_weights = np.mean(weights, axis=0)
    print(f"Averaged weights across {len(weights)} seeds: {mean_weights.shape}")
    return mean_weights


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_importance(importance: dict, output_dir: str, model_key: str):
    """Save concept importance results as JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Full JSON
    json_path = os.path.join(output_dir,
                             f"concept_importance_{model_key}.json")
    with open(json_path, 'w') as f:
        json.dump(importance, f, indent=2)

    # Flat CSV for analysis
    rows = []
    for label, data in importance.items():
        for c in data['positive_concepts']:
            rows.append({
                'label': label,
                'label_readable': data['stats']['label_readable'],
                'direction': 'positive',
                'concept': c['concept'],
                'concept_idx': c['concept_idx'],
                'alignment': c['alignment'],
                'is_core': data['stats']['is_core'],
                'auc': data['stats'].get('auc', np.nan),
            })
        for c in data['negative_concepts']:
            rows.append({
                'label': label,
                'label_readable': data['stats']['label_readable'],
                'direction': 'negative',
                'concept': c['concept'],
                'concept_idx': c['concept_idx'],
                'alignment': c['alignment'],
                'is_core': data['stats']['is_core'],
                'auc': data['stats'].get('auc', np.nan),
            })

    csv_path = os.path.join(output_dir,
                            f"concept_importance_{model_key}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - {os.path.basename(json_path)}")
    print(f"  - {os.path.basename(csv_path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_importance_analysis(args):
    results_dir = args.results_dir

    # Detect embedding model
    model_key = args.embedding_model or detect_embedding_model(results_dir)
    print(f"Embedding model: {model_key}")

    # Load summary for per-label AUCs
    summary = load_summary(results_dir)
    per_label_aucs = {}
    if 'per_label_stats' in summary:
        for label, stats in summary['per_label_stats'].items():
            per_label_aucs[label] = stats['mean']

    # Load model checkpoints
    print(f"\nLoading trained models from {results_dir}...")
    checkpoints = load_trained_models(results_dir, seed=args.seed)
    labels = checkpoints[0]['labels']

    # Get weight matrix (average across seeds)
    if len(checkpoints) == 1:
        model = LogisticRegressionModel(
            checkpoints[0]['model_config']['input_dim'],
            checkpoints[0]['model_config']['output_dim'],
        )
        model.load_state_dict(checkpoints[0]['state_dict'])
        weight_matrix = model.linear.weight.detach().cpu().numpy()
        print(f"Weight matrix from seed {checkpoints[0]['seed']}: "
              f"{weight_matrix.shape}")
    else:
        weight_matrix = aggregate_weights(checkpoints)

    # Load concepts and embeddings
    print()
    concepts, concept_indices = load_concepts()
    concept_embeddings = load_concept_embeddings(model_key, concept_indices, concepts)

    # Verify dimensions match
    assert weight_matrix.shape[1] == concept_embeddings.shape[1], (
        f"Dimension mismatch: weights {weight_matrix.shape[1]} vs "
        f"embeddings {concept_embeddings.shape[1]}"
    )

    # Compute importance
    print(f"\nComputing concept importance (top {args.top_k} per label)...")
    print("=" * 70)
    importance = compute_concept_importance(
        weight_matrix, concept_embeddings,
        labels, concepts, concept_indices,
        top_k=args.top_k,
        per_label_aucs=per_label_aucs,
    )

    # Save
    output_dir = os.path.join(results_dir, 'concept_importance')
    save_importance(importance, output_dir, model_key)

    print(f"\n{'=' * 60}")
    print("CONCEPT IMPORTANCE ANALYSIS COMPLETE")
    print(f"  Embedding model: {model_key}")
    print(f"  Seeds averaged: {len(checkpoints)}")
    print(f"  Labels analyzed: {len(importance)}")
    print(f"  Top-k per direction: {args.top_k}")
    print(f"{'=' * 60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concept importance analysis for MIMIC-CXR "
                    "opportunistic phenotype linear probes")

    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to linear probing results directory '
                             '(e.g., results/linear_mimic_openai_3large)')
    parser.add_argument('--embedding_model', type=str, default=None,
                        choices=list(EMBEDDING_MODELS.keys()),
                        help='Embedding model (auto-detected from dir name)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Specific seed to analyze (default: average all)')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Top-k positive/negative concepts per label')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_importance_analysis(args)
