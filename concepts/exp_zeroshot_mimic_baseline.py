#!/usr/bin/env python3
"""
Baseline zero-shot classification on MIMIC-CXR test set using ViT-7B LoRA CLIP
directly (no concept bottleneck).

Pipeline:
  1. Load finetuned ViT-7B LoRA CLIP model (DINOv3 backbone)
  2. Encode MIMIC-CXR test images via CLIP vision encoder -> image_features [N_images, 768]
  3. Encode pos/neg label prompts via CLIP text encoder -> text_features [N_labels, 768]
  4. Predict: P(label) = sigmoid(cos_sim(image, pos) - cos_sim(image, neg))
  5. Evaluate AUC on 27 phenotype labels

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
from eval import evaluate, bootstrap
import clip as clip_module

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHENOTYPE_LABELS = [
    'heart_failure', 'hfref', 'hfpef', 'atrial_fibrillation',
    'mitral_valve', 'aortic_valve', 'tricuspid_valve', 'pulmonary_valve',
    'cad', 'myocardial_infarction', 'pulmonary_htn', 'stroke',
    'copd', 'asthma', 'ild', 'lung_cancer', 'tuberculosis',
    't2dm', 'obesity', 'dyslipidemia',
    'osteoporosis', 'spinal_stenosis',
    'hypertension', 'ckd', 'pvd', 'liver_cirrhosis', 'anemia',
]

SKIP_EVAL_LABELS = {'pulmonary_valve'}

CORE_PHENOTYPES = [
    'heart_failure', 'hfref', 'hfpef', 'atrial_fibrillation',
    'cad', 'pulmonary_htn', 'copd', 'lung_cancer',
]

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

DEFAULT_MODEL_PATH = (
    "/cbica/projects/CXR/codes/clear_card/checkpoints/"
    "dinov3-vit7b16-lora-r32-a64-combined_vit7b16_20260224_144733/best_model.pt"
)

DEFAULT_LABELS_CSV = os.path.join(os.path.dirname(__file__), '..', 'data',
                                   'mimic_opportunistic_labels.csv')

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

    df = pd.read_csv(args.labels_csv)
    df = df[df['split'] == 'test'].reset_index(drop=True)
    label_cols = [c for c in PHENOTYPE_LABELS if c in df.columns]
    y_true = df[label_cols].values.astype(float)
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_true[y_true < 0] = 0.0
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

    # Encode text prompts
    pos_prompts = [LABEL_TO_PROMPT.get(l, l.replace('_', ' ')) for l in label_cols]
    neg_prompts = [f"no {p}" for p in pos_prompts]

    with torch.no_grad():
        pos_tokens = clip_module.tokenize(pos_prompts, context_length=77).cuda()
        pos_feats = model.encode_text(pos_tokens)
        pos_feats = F.normalize(pos_feats.float(), dim=-1).cpu()

        neg_tokens = clip_module.tokenize(neg_prompts, context_length=77).cuda()
        neg_feats = model.encode_text(neg_tokens)
        neg_feats = F.normalize(neg_feats.float(), dim=-1).cpu()

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

    # Filter evaluable labels
    eval_mask = []
    eval_labels = []
    for i, label in enumerate(label_cols):
        if label in SKIP_EVAL_LABELS:
            eval_mask.append(False)
            continue
        n_pos = int(y_true[:, i].sum())
        if n_pos == 0:
            print(f"  Skipping {label}: 0 test positives")
            eval_mask.append(False)
        else:
            eval_mask.append(True)
            eval_labels.append(label)

    eval_indices = [i for i, m in enumerate(eval_mask) if m]
    y_pred_eval = y_pred[:, eval_indices]
    y_true_eval = y_true[:, eval_indices]

    results_df = evaluate(y_pred_eval, y_true_eval, eval_labels)

    # Per-label AUCs
    aucs = {}
    for label in eval_labels:
        col = f"{label}_auc"
        if col in results_df.columns:
            aucs[label] = float(results_df[col].iloc[0])

    core_aucs = [aucs[c] for c in CORE_PHENOTYPES if c in aucs]
    mean_core_auc = np.mean(core_aucs) if core_aucs else 0.0
    mean_all_auc = np.mean(list(aucs.values())) if aucs else 0.0

    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS (Direct CLIP, no concepts)")
    print(f"{'='*60}")
    print(f"  Mean AUC ({len(core_aucs)} core):  {mean_core_auc:.4f}")
    print(f"  Mean AUC ({len(eval_labels)} eval):  {mean_all_auc:.4f}")
    print(f"  Per-label AUCs:")
    for label in sorted(aucs, key=aucs.get, reverse=True):
        marker = " *" if label in CORE_PHENOTYPES else ""
        print(f"    {label:45s} {aucs[label]:.4f}{marker}")

    # Bootstrap
    cis_df = None
    if args.bootstrap:
        print("\n  Running bootstrap (1000 samples)...")
        boot_stats, cis_df = bootstrap(y_pred_eval, y_true_eval, eval_labels,
                                       n_samples=1000)
        print("  Bootstrap 95% CIs:")
        for label in eval_labels:
            col = f"{label}_auc"
            if col in cis_df.columns:
                lo = cis_df[col]['lower']
                hi = cis_df[col]['upper']
                mn = cis_df[col]['mean']
                print(f"    {label:45s} {mn:.4f} [{lo:.4f}, {hi:.4f}]")

    # Save results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(__file__), 'results',
                            'zeroshot_mimic_baseline_clip')
    os.makedirs(save_dir, exist_ok=True)

    summary = {
        'timestamp': ts,
        'method': 'baseline_clip_zeroshot_mimic (no concepts)',
        'n_test_images': len(y_pred),
        'n_labels': len(label_cols),
        'n_eval_labels': len(eval_labels),
        'labels': label_cols,
        'eval_labels': eval_labels,
        'mean_core_auc': mean_core_auc,
        'mean_all_auc': mean_all_auc,
        'per_label_aucs': aucs,
        'model_path': args.model_path,
        'normalization': {'mean': [101.48761]*3, 'std': [83.43944]*3},
        'resolution': 448,
    }
    with open(os.path.join(save_dir, f"summary_{ts}.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(y_pred, columns=[f"{l}_pred" for l in label_cols]).to_csv(
        os.path.join(save_dir, f"predictions_{ts}.csv"), index=False)
    pd.DataFrame(y_true, columns=[f"{l}_true" for l in label_cols]).to_csv(
        os.path.join(save_dir, f"ground_truth_{ts}.csv"), index=False)
    results_df.to_csv(
        os.path.join(save_dir, f"detailed_aucs_{ts}.csv"), index=False)
    if cis_df is not None:
        cis_df.to_csv(os.path.join(save_dir, f"bootstrap_cis_{ts}.csv"))

    print(f"\n  Results saved to {save_dir}/")
    return aucs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline CLIP zero-shot on MIMIC-CXR (no concept bottleneck)")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--labels_csv', type=str, default=DEFAULT_LABELS_CSV)
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to MIMIC test H5 file')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Disable image feature caching')
    parser.add_argument('--bootstrap', action='store_true', default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline_zeroshot(args)
