#!/usr/bin/env python3
"""
Baseline zero-shot classification on MIMIC-CXR test set using ViT-7B LoRA CLIP
directly (no concept bottleneck).

Pipeline:
  1. Load finetuned ViT-7B LoRA CLIP model (DINOv3 backbone)
  2. Encode MIMIC-CXR test images via CLIP vision encoder -> image_features [N_images, 768]
  3. Encode pos/neg label prompts via CLIP text encoder -> text_features [N_labels, 768]
  4. Predict: P(label) = sigmoid(cos_sim(image, pos) - cos_sim(image, neg))
  5. Evaluate AUC on 1,327 PheWAS phenotype labels

Usage:
  python exp_zeroshot_mimic_baseline.py --h5_path /path/to/mimic_test.h5 [--bootstrap]
"""
import os
import sys
import json
import datetime
import argparse
import hashlib
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from tqdm import tqdm
import h5py

# Add training directory for model loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from zero_shot import load_clip
import clip as clip_module

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Metadata columns in the labels parquet (not phenotype labels)
META_COLUMNS = {'dicom_id', 'subject_id', 'study_id', 'split',
                'ViewPosition', 'icd_source'}

# Min test positives required to evaluate a phenotype
MIN_TEST_POSITIVES = 20

# Phecode info file for phenotype name lookup (used in prompts)
PHECODE_INFO_CSV = os.path.join(os.path.dirname(__file__), '..', 'data',
                                 'mimic_phecode_info.csv')

DEFAULT_MODEL_PATH = (
    "/cbica/projects/CXR/codes/clear_card/checkpoints/"
    "dinov3-vit7b16-lora-r32-a64-combined_vit7b16_20260224_144733/best_model.pt"
)

DEFAULT_LABELS = os.path.join(os.path.dirname(__file__), '..', 'data',
                               'mimic_phewas_labels.parquet')

DINOV3_REPO = "/cbica/projects/CXR/codes/dinov3"
DINOV3_WEIGHTS = "/cbica/projects/CXR/codes/dinov3/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"


def get_cache_tag(model_path: str) -> str:
    """Derive a short hash from the checkpoint directory for cache invalidation."""
    ckpt_dir = os.path.basename(os.path.dirname(os.path.abspath(model_path)))
    return hashlib.md5(ckpt_dir.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MIMICH5Dataset(Dataset):
    def __init__(self, h5_path: str, transform=None):
        self.img_dset = h5py.File(h5_path, 'r')['cxr']
        self.transform = transform
        print(f"MIMICH5Dataset: {len(self.img_dset)} images from {h5_path}")

    def __len__(self):
        return len(self.img_dset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.img_dset[idx]
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        return {'img': img}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_baseline_zeroshot(args):
    print("=" * 70)
    print("Baseline Zero-Shot Classification on MIMIC-CXR Test Set")
    print("  Direct CLIP (no concept bottleneck)")
    print("  Model: DINOv3 ViT-7B + LoRA (r=32, alpha=64)")
    print(f"  H5 path: {args.h5_path}")
    print("=" * 70)

    # Step 1: Load model
    print("\n[Step 1/4] Loading ViT-7B LoRA CLIP model...")
    model = load_clip(
        model_path=args.model_path,
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
        lora_dropout=0.0,
    )
    try:
        backbone = model.visual.backbone
        if hasattr(backbone, 'merge_and_unload'):
            model.visual.backbone = backbone.merge_and_unload()
            print("LoRA weights merged into base model")
    except Exception as e:
        print(f"Warning: could not merge LoRA weights: {e}")
    model = model.cuda().eval()

    # Step 2: Load test data & labels
    print("\n[Step 2/4] Loading test data...")
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761),
                  (83.43944, 83.43944, 83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])
    dataset = MIMICH5Dataset(h5_path=args.h5_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    if args.labels.endswith('.parquet'):
        full_df = pd.read_parquet(args.labels)
    else:
        full_df = pd.read_csv(args.labels)
    df = full_df[full_df['split'] == 'test'].reset_index(drop=True)
    label_cols = [c for c in full_df.columns if c not in META_COLUMNS]
    y_true = df[label_cols].values.astype(np.float32)

    # Load phecode info for phenotype name prompts
    phecode_to_name = {}
    if os.path.exists(PHECODE_INFO_CSV):
        info_df = pd.read_csv(PHECODE_INFO_CSV, dtype={"phecode": str})
        phecode_to_name = dict(zip(info_df['phecode'], info_df['phenotype']))
    print(f"  {len(dataset)} images, {len(label_cols)} labels")

    # Step 3: Encode images and label prompts
    print("\n[Step 3/4] Encoding images and label prompts...")

    # Cache-aware image encoding
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    tag = get_cache_tag(args.model_path)
    cache_path = os.path.join(cache_dir, f"mimic_test_{tag}.pt")

    if os.path.exists(cache_path) and not args.no_cache:
        print(f"  Loading cached test features from {cache_path}")
        image_features = torch.load(cache_path, weights_only=True)
    else:
        image_features = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Encoding images"):
                imgs = batch['img'].cuda()
                feats = model.encode_image(imgs)
                feats = F.normalize(feats.float(), dim=-1)
                image_features.append(feats.cpu())
        image_features = torch.cat(image_features)  # [N_images, 768]
        torch.save(image_features, cache_path)
        print(f"  Cached test features to {cache_path}")
    print(f"  Image features: {image_features.shape}")

    # Encode text prompts — use phenotype names from phecode_info
    pos_prompts = [phecode_to_name.get(l, l.replace('_', ' ')) for l in label_cols]
    neg_prompts = [f"no {p}" for p in pos_prompts]

    # Encode in batches (1,327 labels may not fit in one pass)
    text_batch_size = 256
    print(f"  Encoding {len(pos_prompts)} pos/neg prompt pairs...")
    with torch.no_grad():
        pos_chunks = []
        for i in range(0, len(pos_prompts), text_batch_size):
            tokens = clip_module.tokenize(
                pos_prompts[i:i+text_batch_size], context_length=77).cuda()
            feats = model.encode_text(tokens)
            pos_chunks.append(F.normalize(feats.float(), dim=-1).cpu())
        pos_feats = torch.cat(pos_chunks)

        neg_chunks = []
        for i in range(0, len(neg_prompts), text_batch_size):
            tokens = clip_module.tokenize(
                neg_prompts[i:i+text_batch_size], context_length=77).cuda()
            feats = model.encode_text(tokens)
            neg_chunks.append(F.normalize(feats.float(), dim=-1).cpu())
        neg_feats = torch.cat(neg_chunks)

    print(f"  Pos text features: {pos_feats.shape}")
    print(f"  Neg text features: {neg_feats.shape}")
    print(f"  Sample: pos='{pos_prompts[0]}', neg='{neg_prompts[0]}'")

    # Step 4: Predict & evaluate
    print("\n[Step 4/4] Computing predictions and evaluating...")
    with torch.no_grad():
        logits_pos = image_features @ pos_feats.T  # [N_images, N_labels]
        logits_neg = image_features @ neg_feats.T
        probs = torch.sigmoid(logits_pos - logits_neg)
        y_pred = probs.numpy()

    # Filter evaluable labels (need >= MIN_TEST_POSITIVES)
    from sklearn.metrics import roc_auc_score
    eval_labels = []
    eval_indices = []
    for i, label in enumerate(label_cols):
        n_pos = int(y_true[:, i].sum())
        if n_pos >= MIN_TEST_POSITIVES and n_pos < len(y_true):
            eval_labels.append(label)
            eval_indices.append(i)

    y_pred_eval = y_pred[:, eval_indices]
    y_true_eval = y_true[:, eval_indices]

    # Per-label AUCs
    aucs = {}
    for j, label in enumerate(eval_labels):
        aucs[label] = roc_auc_score(y_true_eval[:, j], y_pred_eval[:, j])

    mean_all_auc = np.mean(list(aucs.values())) if aucs else 0.0

    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS (Direct CLIP, no concepts)")
    print(f"{'='*60}")
    print(f"  Macro AUROC ({len(eval_labels)} phenotypes):  {mean_all_auc:.4f}")
    sorted_aucs = sorted(aucs.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 5:")
    for label, auc_val in sorted_aucs[:5]:
        name = phecode_to_name.get(label, label)
        print(f"    {label:10s} ({name:40s}) {auc_val:.4f}")
    print(f"  Bottom 5:")
    for label, auc_val in sorted_aucs[-5:]:
        name = phecode_to_name.get(label, label)
        print(f"    {label:10s} ({name:40s}) {auc_val:.4f}")

    # Bootstrap CIs on macro AUROC
    if args.bootstrap:
        print("\n  Running bootstrap (1000 samples) on macro AUROC...")
        n_boot = 1000
        rng = np.random.RandomState(42)
        boot_macro = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(y_true_eval), len(y_true_eval))
            boot_aucs = []
            for j in range(len(eval_labels)):
                if y_true_eval[idx, j].sum() > 0:
                    boot_aucs.append(roc_auc_score(
                        y_true_eval[idx, j], y_pred_eval[idx, j]))
            if boot_aucs:
                boot_macro.append(np.mean(boot_aucs))
        lo, hi = np.percentile(boot_macro, [2.5, 97.5])
        print(f"  Macro AUROC 95% CI: [{lo:.4f}, {hi:.4f}]")

    # Save results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(__file__), 'results',
                            'zeroshot_mimic_baseline_clip')
    os.makedirs(save_dir, exist_ok=True)

    summary = {
        'timestamp': ts,
        'method': 'baseline_clip_zeroshot_mimic (no concepts)',
        'n_test_images': len(y_pred),
        'n_phenotypes_total': len(label_cols),
        'n_phenotypes_eval': len(eval_labels),
        'macro_auroc': mean_all_auc,
        'per_label_aucs': aucs,
        'model_path': args.model_path,
        'normalization': {'mean': [101.48761]*3, 'std': [83.43944]*3},
        'resolution': 448,
    }
    with open(os.path.join(save_dir, f"summary_{ts}.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save per-label AUC table
    auc_rows = [{'phecode': l, 'phenotype': phecode_to_name.get(l, l),
                 'auc': v} for l, v in aucs.items()]
    pd.DataFrame(auc_rows).to_csv(
        os.path.join(save_dir, f"per_label_aucs_{ts}.csv"), index=False)

    print(f"\n  Results saved to {save_dir}/")
    return aucs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline CLIP zero-shot on MIMIC-CXR (no concept bottleneck)")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--labels', type=str, default=DEFAULT_LABELS)
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to MIMIC test H5 file')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Disable image feature caching')
    parser.add_argument('--bootstrap', action='store_true', default=False,
                        help='Run bootstrap for macro AUROC confidence interval')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline_zeroshot(args)
