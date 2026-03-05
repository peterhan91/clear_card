#!/usr/bin/env python3
"""
Concept importance visualization for MIMIC-CXR opportunistic phenotype
linear probes.

Loads per-seed model checkpoints and concept embeddings, computes concept
importance (cosine similarity between linear weights and LLM embeddings)
for each seed, then creates publication-quality horizontal bar charts with:
  - Top 10 positive concepts per phenotype
  - Error bars (std across seeds)
  - Individual seed data points as open circles

Adapted from cxr_concept/concepts/plot_linear_probing_concepts.py for the
CLEAR pipeline with 27 MIMIC-CXR phenotypes and 5 seeds.

Usage:
  python plot_concept_importance.py
  python plot_concept_importance.py --results_dir results/linear_mimic_kalm_gemma3_12b
  python plot_concept_importance.py --labels heart_failure copd lung_cancer
  python plot_concept_importance.py --top_k 15 --all_labels
"""

import os
import sys
import glob
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory for model imports
sys.path.insert(0, os.path.dirname(__file__))
from exp_linear_mimic import (
    LogisticRegressionModel, EMBEDDING_MODELS, EMBEDDINGS_DIR,
    CONCEPTS_CSV, SKIP_EVAL_LABELS, CORE_PHENOTYPES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_DISPLAY = {
    'heart_failure': 'Heart Failure',
    'hfref': 'HFrEF',
    'hfpef': 'HFpEF',
    'atrial_fibrillation': 'Atrial Fibrillation',
    'mitral_valve': 'Mitral Valve Disease',
    'aortic_valve': 'Aortic Valve Disease',
    'tricuspid_valve': 'Tricuspid Valve Disease',
    'cad': 'Coronary Artery Disease',
    'myocardial_infarction': 'Myocardial Infarction',
    'pulmonary_htn': 'Pulmonary Hypertension',
    'stroke': 'Stroke',
    'copd': 'COPD',
    'asthma': 'Asthma',
    'ild': 'Interstitial Lung Disease',
    'lung_cancer': 'Lung Cancer',
    'tuberculosis': 'Tuberculosis',
    't2dm': 'Type 2 Diabetes',
    'obesity': 'Obesity',
    'dyslipidemia': 'Dyslipidemia',
    'osteoporosis': 'Osteoporosis',
    'spinal_stenosis': 'Spinal Stenosis',
    'hypertension': 'Hypertension',
    'ckd': 'Chronic Kidney Disease',
    'pvd': 'Peripheral Vascular Disease',
    'liver_cirrhosis': 'Liver Cirrhosis',
    'anemia': 'Anemia',
}

COLOR_POS = '#CD5C5C'  # Indian red


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_concepts_and_embeddings(model_key):
    """Load concept texts and their LLM embeddings."""
    print("Loading concepts and embeddings...")
    df = pd.read_csv(CONCEPTS_CSV)
    concepts = df['concept'].tolist()
    concept_indices = df['concept_idx'].tolist()

    info = EMBEDDING_MODELS[model_key]
    pickle_path = os.path.join(EMBEDDINGS_DIR,
                               f"cxr_embeddings_{info['pickle_suffix']}.pickle")
    print(f"  Embeddings: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    first_key = next(iter(data))
    uses_string_keys = isinstance(first_key, str)
    if uses_string_keys:
        data_lower = {k.lower(): v for k, v in data.items()}
    else:
        data_lower = None

    dim = info['dim']
    embeddings = np.zeros((len(concept_indices), dim), dtype=np.float32)
    missing = 0
    for pos, idx in enumerate(concept_indices):
        emb = None
        if not uses_string_keys:
            emb = data.get(idx)
        else:
            text = concepts[pos]
            emb = data.get(text)
            if emb is None:
                emb = data_lower.get(text.lower())
        if emb is not None:
            embeddings[pos] = np.array(emb, dtype=np.float32)
        else:
            missing += 1

    if missing:
        print(f"  Warning: {missing}/{len(concept_indices)} concepts missing")
    else:
        print(f"  All {len(concept_indices)} concepts matched")

    concept_embeddings = torch.tensor(embeddings).float()
    print(f"  Loaded {len(concepts)} concepts with {dim}-dim embeddings")
    return concepts, concept_indices, concept_embeddings


def detect_embedding_model(results_dir):
    """Detect which embedding model was used from the results directory name."""
    dirname = os.path.basename(results_dir)
    for key in EMBEDDING_MODELS:
        if key in dirname:
            return key
    raise ValueError(f"Cannot detect embedding model from directory: {dirname}")


def load_per_seed_importance(results_dir, concepts, concept_embeddings):
    """Load linear probe weights from all seeds and compute per-seed concept importance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    concept_embeddings_gpu = concept_embeddings.to(device)
    concept_embeddings_norm = torch.nn.functional.normalize(concept_embeddings_gpu, dim=1)

    models_dir = os.path.join(results_dir, 'models')
    paths = sorted(glob.glob(os.path.join(models_dir, "seed_*_model.pth")))
    if not paths:
        raise FileNotFoundError(f"No model checkpoints in {models_dir}")

    all_importance = []  # [n_seeds, n_concepts, n_labels]
    labels = None

    for path in tqdm(paths, desc="Loading seeds"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if labels is None:
            labels = ckpt['labels']

        model = LogisticRegressionModel(
            ckpt['model_config']['input_dim'],
            ckpt['model_config']['output_dim'],
        )
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(device)

        # [n_labels, emb_dim]
        weights = model.linear.weight.detach()
        weights_norm = torch.nn.functional.normalize(weights, dim=1)

        # Cosine similarity: [n_concepts, n_labels]
        importance = torch.matmul(concept_embeddings_norm, weights_norm.T)
        all_importance.append(importance.cpu().numpy())

        del model, weights, weights_norm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_importance = np.array(all_importance)  # [n_seeds, n_concepts, n_labels]
    avg = np.mean(all_importance, axis=0)
    std = np.std(all_importance, axis=0)

    print(f"Computed importance: {all_importance.shape[0]} seeds, "
          f"{all_importance.shape[1]} concepts, {all_importance.shape[2]} labels")
    return avg, std, all_importance, labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def wrap_label(text, max_words=5):
    """Capitalize and wrap long concept labels."""
    text = text.strip().capitalize()
    words = text.split()
    if len(words) > max_words:
        lines = []
        for i in range(0, len(words), max_words):
            lines.append(' '.join(words[i:i + max_words]))
        return '\n'.join(lines)
    return text


def plot_single_label(avg_weights, std_weights, all_weights, concepts,
                      label, label_idx, top_k=10, output_dir=None):
    """Create a horizontal bar chart for one phenotype with error bars and seed points."""
    label_avg = avg_weights[:, label_idx]
    label_std = std_weights[:, label_idx]
    label_all = all_weights[:, :, label_idx]  # [n_seeds, n_concepts]

    # Get top-k positive concepts
    df = pd.DataFrame({
        'concept': concepts,
        'weight': label_avg,
        'std': label_std,
        'concept_idx': range(len(concepts)),
    })
    df_pos = df[df['weight'] > 0].sort_values('weight', ascending=False).head(top_k)

    if len(df_pos) == 0:
        print(f"  {label}: no positive concepts, skipping")
        return

    combined_concepts = df_pos['concept'].tolist()
    combined_weights = df_pos['weight'].tolist()
    combined_stds = df_pos['std'].tolist()
    combined_indices = df_pos['concept_idx'].tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(13, 12))

    y_positions = list(range(len(combined_weights)))
    bars = ax.barh(y_positions, combined_weights, xerr=combined_stds,
                   color=COLOR_POS, alpha=0.8, edgecolor='black', linewidth=1,
                   capsize=0, error_kw={'linewidth': 1.5, 'alpha': 0.8})

    # Individual seed data points as open circles
    rng = np.random.RandomState(42)
    for i, c_idx in enumerate(combined_indices):
        seed_values = label_all[:, c_idx]
        y_jitter = rng.normal(y_positions[i], 0.05, len(seed_values))
        ax.scatter(seed_values, y_jitter,
                   s=30, facecolors='none', edgecolors='black',
                   alpha=0.6, linewidth=1)

    # Labels
    clean_labels = [wrap_label(c) for c in combined_concepts]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(clean_labels, fontsize=18)
    ax.tick_params(axis='x', labelsize=24)
    ax.set_xlabel('Concept importance score', fontsize=28)
    ax.invert_yaxis()

    # Adaptive x-axis: zoom to range of plotted data
    all_vals = combined_weights.copy()
    for c_idx in combined_indices:
        all_vals.extend(label_all[:, c_idx].tolist())
    min_val = min(combined_weights)
    max_val = max(all_vals)
    padding = (max_val - min_val) * 0.05
    ax.set_xlim(min_val - padding, max_val + padding)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='x')

    display = LABEL_DISPLAY.get(label, label)
    ax.set_title(display, fontsize=32, pad=10)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe = label.replace(' ', '_')
        for ext in ['png', 'pdf']:
            path = os.path.join(output_dir, f"concept_importance_{safe}.{ext}")
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: concept_importance_{safe}.{{png,pdf}}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Plot concept importance for MIMIC phenotype linear probes')
    parser.add_argument(
        '--results_dir', type=str,
        default=os.path.join(os.path.dirname(__file__),
                             'results', 'linear_mimic_kalm_gemma3_12b'),
        help='Linear probing results directory')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Specific labels to plot (default: core phenotypes)')
    parser.add_argument('--all_labels', action='store_true',
                        help='Plot all labels')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top positive concepts to show')
    args = parser.parse_args()

    # Detect embedding model
    model_key = detect_embedding_model(args.results_dir)
    print(f"Embedding model: {model_key}")

    # Load concepts and embeddings
    concepts, concept_indices, concept_embeddings = load_concepts_and_embeddings(model_key)

    # Load per-seed importance
    avg_weights, std_weights, all_weights, labels = load_per_seed_importance(
        args.results_dir, concepts, concept_embeddings)

    # Determine which labels to plot
    eval_labels = [l for l in labels if l not in SKIP_EVAL_LABELS]
    if args.all_labels:
        plot_labels = eval_labels
    elif args.labels:
        plot_labels = [l for l in args.labels if l in eval_labels]
    else:
        plot_labels = [l for l in CORE_PHENOTYPES if l in eval_labels]

    print(f"\nPlotting {len(plot_labels)} labels: {plot_labels}")

    output_dir = os.path.join(args.results_dir, 'concept_importance', 'plots')

    for label in plot_labels:
        label_idx = labels.index(label)
        plot_single_label(avg_weights, std_weights, all_weights, concepts,
                          label, label_idx, top_k=args.top_k,
                          output_dir=output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
