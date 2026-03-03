#!/usr/bin/env python3
"""
Zero-shot concept-based classification on CheXpert test set.

Pipeline:
  1. Load finetuned ViT-7B LoRA CLIP model (DINOv3 backbone)
  2. Merge LoRA weights into base model for efficient inference
  3. Load ~492k radiological concepts and their precomputed LLM embeddings
  4. Encode concepts via CLIP text encoder -> concept_features  [N_concepts, 768]
  5. Encode CheXpert test images via CLIP vision encoder -> image_features  [N_images, 768]
  6. Concept similarity: scores = image_features @ concept_features.T  [N_images, N_concepts]
  7. Project to LLM space: llm_repr = scores @ concept_embeddings  [N_images, emb_dim]
  8. Embed pos/neg class prompts in same LLM space  [N_labels, emb_dim] x 2
  9. Predict: P(label) = softmax(cos_sim(llm_repr, pos), cos_sim(llm_repr, neg))
 10. Evaluate AUC on 14 CheXpert labels

Usage:
  python exp_zeroshot.py [--embedding_models sfr_mistral nemotron_8b kalm_gemma3_12b]
                         [--model_path /path/to/best_model.pt]
                         [--max_concepts 0]  # 0 = use all
"""
import os
import sys
import json
import pickle
import datetime
import argparse
import gc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from tqdm import tqdm

# Add training directory for model loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from zero_shot import load_clip, CXRTestDataset
from eval import evaluate, bootstrap, compute_cis
import clip as clip_module  # CLIP tokenizer

# Add current directory for embedding utilities
from get_embed import RadiologyEmbeddingGenerator, MODEL_CONFIGS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices',
]

CORE_CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion',
]

# Best ViT-7B LoRA model (combined CheXpert+ReXGradient)
DEFAULT_MODEL_PATH = (
    "/cbica/projects/CXR/codes/clear_card/checkpoints/"
    "dinov3-vit7b16-lora-r32-a64-combined_vit7b16_20260224_144733/best_model.pt"
)
DEFAULT_CONFIG_PATH = (
    "/cbica/projects/CXR/codes/clear_card/checkpoints/"
    "dinov3-vit7b16-lora-r32-a64-combined_vit7b16_20260224_144733/config.json"
)

# Data paths
CHEXPERT_TEST_H5 = "/cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_test.h5"
CHEXPERT_TEST_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'chexpert_test.csv')

# Concepts
CONCEPTS_CSV = os.path.join(os.path.dirname(__file__), 'concepts_clean.csv')
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), 'embeddings_output')

# DINOv3
DINOV3_REPO = "/cbica/projects/CXR/codes/dinov3"
DINOV3_WEIGHTS = "/cbica/projects/CXR/codes/dinov3/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"

# Embedding model mapping: short name -> (HuggingFace name, pickle suffix, embedding dim)
EMBEDDING_MODELS = {
    'sfr_mistral': {
        'hf_name': 'Salesforce/SFR-Embedding-Mistral',
        'pickle_suffix': 'sfr_mistral',
        'dim': 4096,
    },
    'nemotron_8b': {
        'hf_name': 'nvidia/llama-embed-nemotron-8b',
        'pickle_suffix': 'nemotron_8b',
        'dim': 4096,
    },
    'kalm_gemma3_12b': {
        'hf_name': 'tencent/KaLM-Embedding-Gemma3-12B-2511',
        'pickle_suffix': 'kalm_gemma3_12b',
        'dim': 3840,
    },
}


def print_gpu_memory(stage=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        res = torch.cuda.memory_reserved() / 1024**3
        print(f"  [GPU] {stage}: {alloc:.2f} GB allocated, {res:.2f} GB reserved")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, merge_lora: bool = True) -> torch.nn.Module:
    """Load the ViT-7B LoRA CLIP model and optionally merge LoRA weights."""
    print(f"Loading model from {model_path}")
    print_gpu_memory("before model load")

    model = load_clip(
        model_path=model_path,
        pretrained=False,
        context_length=77,
        use_dinov3=True,
        dinov3_model_name="dinov3_vit7b16",
        dinov3_repo_dir=DINOV3_REPO,
        dinov3_weights=DINOV3_WEIGHTS,
        freeze_dinov3=False,
        use_lora=True,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.0,  # no dropout at inference
    )

    # Merge LoRA weights into the base model for cleaner/faster inference
    if merge_lora:
        try:
            backbone = model.visual.backbone
            if hasattr(backbone, 'merge_and_unload'):
                model.visual.backbone = backbone.merge_and_unload()
                print("LoRA weights merged into base model")
        except Exception as e:
            print(f"Warning: could not merge LoRA weights: {e}")

    model = model.cuda().eval()
    print_gpu_memory("after model load")
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_test_data(batch_size: int = 32):
    """Load CheXpert test dataset and ground-truth labels."""
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761),
                  (83.43944,  83.43944,  83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])

    dataset = CXRTestDataset(img_path=CHEXPERT_TEST_H5, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    # Ground-truth labels
    df = pd.read_csv(CHEXPERT_TEST_CSV)
    exclude = {'Study', 'Path', 'image_id', 'ImageID', 'filename'}
    label_cols = [c for c in CHEXPERT_LABELS if c in df.columns]
    y_true = df[label_cols].values.astype(float)
    # Replace NaN / -1 with 0 (treat uncertain as negative)
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_true[y_true < 0] = 0.0

    print(f"Test set: {len(dataset)} images, {y_true.shape[1]} labels")
    return loader, y_true, label_cols


# ---------------------------------------------------------------------------
# Concept loading
# ---------------------------------------------------------------------------
def load_concepts(max_concepts: int = 0):
    """Load cleaned concepts from CSV."""
    df = pd.read_csv(CONCEPTS_CSV)
    concepts = df['concept'].tolist()
    indices = df['concept_idx'].tolist()
    if max_concepts > 0:
        concepts = concepts[:max_concepts]
        indices = indices[:max_concepts]
    print(f"Loaded {len(concepts)} concepts (top-5: {concepts[:5]})")
    return concepts, indices


def load_concept_embeddings(model_key: str, concept_indices: List[int]):
    """Load precomputed concept embeddings from pickle and align to concept order."""
    info = EMBEDDING_MODELS[model_key]
    pickle_path = os.path.join(EMBEDDINGS_DIR,
                               f"cxr_embeddings_{info['pickle_suffix']}.pickle")
    print(f"Loading concept embeddings: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    dim = info['dim']
    embeddings = np.zeros((len(concept_indices), dim), dtype=np.float32)
    missing = 0
    for pos, idx in enumerate(concept_indices):
        if idx in data:
            emb = data[idx]
            if isinstance(emb, np.ndarray):
                embeddings[pos] = emb.astype(np.float32)
            else:
                embeddings[pos] = np.array(emb, dtype=np.float32)
        else:
            missing += 1

    if missing:
        print(f"  Warning: {missing}/{len(concept_indices)} concepts missing embeddings")
    else:
        print(f"  All {len(concept_indices)} concepts matched")

    print(f"  Embeddings shape: {embeddings.shape}")
    return torch.from_numpy(embeddings)  # [N_concepts, dim]


# ---------------------------------------------------------------------------
# CLIP encoding
# ---------------------------------------------------------------------------
def encode_concepts(model, concepts: List[str], batch_size: int = 512):
    """Encode concepts through CLIP text encoder."""
    all_features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(concepts), batch_size), desc="Encoding concepts"):
            batch = concepts[i:i + batch_size]
            tokens = clip_module.tokenize(batch, context_length=77).cuda()
            feats = model.encode_text(tokens)
            feats = F.normalize(feats.float(), dim=-1)
            all_features.append(feats.cpu())
            torch.cuda.empty_cache()

    return torch.cat(all_features)  # [N_concepts, 768]


def encode_images(model, loader):
    """Encode test images through CLIP vision encoder."""
    all_features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding images"):
            imgs = batch['img'].cuda()
            feats = model.encode_image(imgs)
            feats = F.normalize(feats.float(), dim=-1)
            all_features.append(feats.cpu())

    return torch.cat(all_features)  # [N_images, 768]


# ---------------------------------------------------------------------------
# Zero-shot prediction
# ---------------------------------------------------------------------------
def compute_concept_scores(image_features: torch.Tensor,
                           concept_features: torch.Tensor,
                           batch_size: int = 100):
    """Compute cosine similarity between images and concepts.
    Returns [N_images, N_concepts] on CPU."""
    chunks = []
    for i in range(0, len(image_features), batch_size):
        img_batch = image_features[i:i + batch_size].cuda()
        cf = concept_features.cuda()
        sim = img_batch @ cf.T
        chunks.append(sim.cpu())
        torch.cuda.empty_cache()
    return torch.cat(chunks)  # [N_images, N_concepts]


def project_to_llm_space(concept_scores: torch.Tensor,
                          concept_embeddings: torch.Tensor):
    """Project images into LLM embedding space via concept scores.
    concept_scores: [N_images, N_concepts]
    concept_embeddings: [N_concepts, emb_dim]
    Returns normalized [N_images, emb_dim] on GPU.
    """
    # Do matmul on GPU in chunks to avoid OOM
    batch_size = 64
    llm_repr_chunks = []
    ce = concept_embeddings.cuda().float()
    for i in range(0, len(concept_scores), batch_size):
        cs_batch = concept_scores[i:i + batch_size].cuda().float()
        llm_batch = cs_batch @ ce
        # Clamp to avoid inf/nan from large values
        llm_batch = torch.clamp(llm_batch, min=-1e6, max=1e6)
        norms = llm_batch.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        llm_batch = llm_batch / norms
        llm_repr_chunks.append(llm_batch.cpu())
        torch.cuda.empty_cache()

    llm_repr = torch.cat(llm_repr_chunks)
    return llm_repr  # [N_images, emb_dim]


def generate_prompt_embeddings(model_key: str, labels: List[str]):
    """Generate LLM embeddings for positive/negative prompts using the
    same embedding model that was used for concept embeddings.

    Returns pos_embeds [N_labels, dim], neg_embeds [N_labels, dim] as tensors.
    """
    info = EMBEDDING_MODELS[model_key]
    hf_name = info['hf_name']

    pos_prompts = [label.lower() for label in labels]
    neg_prompts = [f"no {label.lower()}" for label in labels]
    all_prompts = pos_prompts + neg_prompts

    print(f"  Generating {len(all_prompts)} prompt embeddings with {hf_name}")

    # Use sentence_transformers type which handles MODEL_CONFIGS properly
    generator = RadiologyEmbeddingGenerator(
        embedding_type="sentence_transformers",
        local_model_name=hf_name,
        batch_size=len(all_prompts),  # all prompts in one batch
        use_fp16=True,
        trust_remote_code=True,
    )

    embeddings = generator.get_sentence_transformer_embeddings_batch(all_prompts)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Cleanup the embedding model to free GPU memory
    del generator
    torch.cuda.empty_cache()
    gc.collect()

    n = len(labels)
    pos_embeds = torch.from_numpy(embeddings[:n])
    neg_embeds = torch.from_numpy(embeddings[n:])

    # Normalize
    pos_embeds = F.normalize(pos_embeds.float(), dim=-1)
    neg_embeds = F.normalize(neg_embeds.float(), dim=-1)

    return pos_embeds, neg_embeds


def predict(llm_repr: torch.Tensor,
            pos_embeds: torch.Tensor,
            neg_embeds: torch.Tensor):
    """Compute softmax predictions from LLM-space representations.
    Returns y_pred [N_images, N_labels] as numpy.
    """
    with torch.no_grad():
        llm_gpu = llm_repr.cuda().float()
        pos_gpu = pos_embeds.cuda().float()
        neg_gpu = neg_embeds.cuda().float()

        logits_pos = llm_gpu @ pos_gpu.T  # [N_images, N_labels]
        logits_neg = llm_gpu @ neg_gpu.T

        # Numerically stable: P = sigmoid(pos - neg)
        probs = torch.sigmoid(logits_pos - logits_neg)

        y_pred = probs.cpu().numpy()

    torch.cuda.empty_cache()
    return y_pred


# ---------------------------------------------------------------------------
# Evaluation & saving
# ---------------------------------------------------------------------------
def evaluate_and_report(y_pred: np.ndarray, y_true: np.ndarray,
                        labels: List[str], model_key: str,
                        n_concepts: int, run_bootstrap: bool = False):
    """Evaluate predictions, print results, return metrics dict."""
    results_df = evaluate(y_pred, y_true, labels)

    # Per-label AUCs
    aucs = {}
    for label in labels:
        col = f"{label}_auc"
        if col in results_df.columns:
            aucs[label] = float(results_df[col].iloc[0])

    # Core-condition mean AUC
    core_aucs = [aucs[c] for c in CORE_CONDITIONS if c in aucs]
    mean_core_auc = np.mean(core_aucs) if core_aucs else 0.0
    mean_all_auc = np.mean(list(aucs.values())) if aucs else 0.0

    print(f"\n{'='*60}")
    print(f"Results for {model_key.upper()} ({n_concepts} concepts)")
    print(f"{'='*60}")
    print(f"  Mean AUC (5 core):  {mean_core_auc:.4f}")
    print(f"  Mean AUC (all 14):  {mean_all_auc:.4f}")
    print(f"  Per-label AUCs:")
    for label, auc_val in aucs.items():
        marker = " *" if label in CORE_CONDITIONS else ""
        print(f"    {label:35s} {auc_val:.4f}{marker}")

    # Bootstrap CIs
    cis_df = None
    if run_bootstrap:
        print("\n  Running bootstrap (1000 samples)...")
        boot_stats, cis_df = bootstrap(y_pred, y_true, labels, n_samples=1000)
        print("  Bootstrap 95% CIs:")
        for label in labels:
            col = f"{label}_auc"
            if col in cis_df.columns:
                lo = cis_df[col]['lower']
                hi = cis_df[col]['upper']
                mn = cis_df[col]['mean']
                print(f"    {label:35s} {mn:.4f} [{lo:.4f}, {hi:.4f}]")

    return {
        'model': model_key,
        'n_concepts': n_concepts,
        'mean_core_auc': mean_core_auc,
        'mean_all_auc': mean_all_auc,
        'per_label_aucs': aucs,
        'results_df': results_df,
        'cis_df': cis_df,
    }


def save_results(metrics: dict, y_pred: np.ndarray, y_true: np.ndarray,
                 labels: List[str], output_dir: str):
    """Save evaluation results to disk."""
    model_key = metrics['model']
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"zeroshot_{model_key}")
    os.makedirs(save_dir, exist_ok=True)

    # Summary JSON
    summary = {
        'timestamp': ts,
        'method': 'concept_zeroshot_vit7b_lora',
        'embedding_model': model_key,
        'n_concepts': metrics['n_concepts'],
        'n_test_images': len(y_pred),
        'n_labels': len(labels),
        'labels': labels,
        'mean_core_auc': metrics['mean_core_auc'],
        'mean_all_auc': metrics['mean_all_auc'],
        'per_label_aucs': metrics['per_label_aucs'],
        'model_path': DEFAULT_MODEL_PATH,
        'normalization': {'mean': [101.48761]*3, 'std': [83.43944]*3},
        'resolution': 448,
    }
    with open(os.path.join(save_dir, f"summary_{ts}.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # Predictions & ground truth
    pd.DataFrame(y_pred, columns=[f"{l}_pred" for l in labels]).to_csv(
        os.path.join(save_dir, f"predictions_{ts}.csv"), index=False)
    pd.DataFrame(y_true, columns=[f"{l}_true" for l in labels]).to_csv(
        os.path.join(save_dir, f"ground_truth_{ts}.csv"), index=False)

    # Detailed AUCs
    metrics['results_df'].to_csv(
        os.path.join(save_dir, f"detailed_aucs_{ts}.csv"), index=False)

    if metrics['cis_df'] is not None:
        metrics['cis_df'].to_csv(
            os.path.join(save_dir, f"bootstrap_cis_{ts}.csv"))

    print(f"  Results saved to {save_dir}/")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_zeroshot_pipeline(args):
    """Run the full concept-based zero-shot classification pipeline."""

    print("=" * 70)
    print("Concept-Based Zero-Shot Classification on CheXpert Test Set")
    print("  Model: DINOv3 ViT-7B + LoRA (r=32, alpha=64)")
    print(f"  Embedding models: {args.embedding_models}")
    print("=" * 70)

    # Step 1: Load CLIP model
    print("\n[Step 1/6] Loading ViT-7B LoRA CLIP model...")
    model = load_model(args.model_path, merge_lora=args.merge_lora)

    # Step 2: Load test data
    print("\n[Step 2/6] Loading CheXpert test data...")
    loader, y_true, label_cols = load_test_data(batch_size=args.image_batch_size)

    # Step 3: Load concepts
    print("\n[Step 3/6] Loading concepts...")
    concepts, concept_indices = load_concepts(max_concepts=args.max_concepts)

    # Step 4: Encode concepts & images through CLIP
    print("\n[Step 4/6] Encoding concepts and images through CLIP...")
    concept_features = encode_concepts(model, concepts,
                                       batch_size=args.concept_batch_size)
    image_features = encode_images(model, loader)

    # Compute concept similarity scores (reused across embedding models)
    print("\n  Computing concept similarity scores...")
    concept_scores = compute_concept_scores(image_features, concept_features,
                                            batch_size=args.sim_batch_size)
    print(f"  Concept scores shape: {concept_scores.shape}")

    # Free CLIP model to make room for embedding models
    print("\n  Freeing CLIP model from GPU...")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print_gpu_memory("after freeing CLIP")

    # Step 5-6: For each embedding model, project & classify
    all_metrics = {}
    output_dir = os.path.join(os.path.dirname(__file__), 'results')

    for model_key in args.embedding_models:
        print(f"\n{'='*70}")
        print(f"[Step 5/6] Processing embedding model: {model_key.upper()}")
        print(f"{'='*70}")

        # Load precomputed concept embeddings
        concept_embeds = load_concept_embeddings(model_key, concept_indices)

        # Project images to LLM space
        print("  Projecting images to LLM embedding space...")
        llm_repr = project_to_llm_space(concept_scores, concept_embeds)
        print(f"  LLM representation shape: {llm_repr.shape}")

        # Generate prompt embeddings (loads embedding model temporarily)
        print("  Generating class prompt embeddings...")
        pos_embeds, neg_embeds = generate_prompt_embeddings(model_key, label_cols)
        print(f"  Pos embeddings: {pos_embeds.shape}, Neg: {neg_embeds.shape}")

        # Predict
        print("\n[Step 6/6] Computing predictions...")
        y_pred = predict(llm_repr, pos_embeds, neg_embeds)
        print(f"  Predictions shape: {y_pred.shape}")

        # Evaluate
        metrics = evaluate_and_report(
            y_pred, y_true, label_cols, model_key,
            n_concepts=len(concepts),
            run_bootstrap=args.bootstrap,
        )
        all_metrics[model_key] = metrics

        # Save
        save_results(metrics, y_pred, y_true, label_cols, output_dir)

        # Cleanup
        del concept_embeds, llm_repr, pos_embeds, neg_embeds, y_pred
        torch.cuda.empty_cache()
        gc.collect()

    # Final comparison summary
    print(f"\n{'='*70}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':25s} {'Core AUC':>10s} {'All AUC':>10s}")
    print("-" * 50)
    for model_key, m in all_metrics.items():
        print(f"{model_key:25s} {m['mean_core_auc']:10.4f} {m['mean_all_auc']:10.4f}")

    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot concept-based classification on CheXpert test set")

    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to best_model.pt checkpoint')
    parser.add_argument('--embedding_models', nargs='+',
                        default=['sfr_mistral', 'nemotron_8b', 'kalm_gemma3_12b'],
                        choices=list(EMBEDDING_MODELS.keys()),
                        help='Embedding models to evaluate')
    parser.add_argument('--max_concepts', type=int, default=0,
                        help='Max concepts to use (0 = all ~492k)')
    parser.add_argument('--image_batch_size', type=int, default=16,
                        help='Batch size for image encoding (ViT-7B is large)')
    parser.add_argument('--concept_batch_size', type=int, default=512,
                        help='Batch size for concept text encoding')
    parser.add_argument('--sim_batch_size', type=int, default=64,
                        help='Batch size for similarity computation')
    parser.add_argument('--merge_lora', action='store_true', default=True,
                        help='Merge LoRA weights into base model')
    parser.add_argument('--no_merge_lora', dest='merge_lora', action='store_false',
                        help='Keep LoRA weights separate')
    parser.add_argument('--bootstrap', action='store_true', default=False,
                        help='Run bootstrap for confidence intervals')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_zeroshot_pipeline(args)
