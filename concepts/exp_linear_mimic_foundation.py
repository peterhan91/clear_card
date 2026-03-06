#!/usr/bin/env python3
"""
Foundation Model Linear Probing on MIMIC-CXR
1,327 PheWAS Phenotypes (following Merlin's full phenotype evaluation)

Extracts image features from pretrained foundation models and trains
linear probes.  No concept-based projection -- features are used directly.

Models supported:
  ark_plus    - Ark+ Swin-Large-768             (1376-d after projector)
  rad_dino    - RAD-DINO ViT-B/14               (768-d CLS token)
  chexzero    - CheXzero CLIP ViT-B/32          (512-d)
  biomedclip  - BiomedCLIP ViT-B/16+PubMedBERT  (512-d)

Usage:
  python exp_linear_mimic_foundation.py --model rad_dino
  python exp_linear_mimic_foundation.py --model ark_plus \
      --ark_checkpoint /path/to/Ark6_swinLarge768_ep50.pth.tar
  python exp_linear_mimic_foundation.py --model chexzero \
      --chexzero_checkpoint /path/to/best_64_5e-05_original_22000_0.864.pt
  python exp_linear_mimic_foundation.py --model biomedclip
  python exp_linear_mimic_foundation.py --model all   # run all (needs all checkpoints)
"""

import os
import json
import pickle
import datetime
import argparse
import gc
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants  (shared with exp_linear_mimic.py)
# ---------------------------------------------------------------------------
# Metadata columns in the labels parquet (not phenotype labels)
META_COLUMNS = {'dicom_id', 'subject_id', 'study_id', 'split',
                'ViewPosition', 'icd_source'}

# Min test positives required to evaluate a phenotype
MIN_TEST_POSITIVES = 20

DEFAULT_LABELS = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'mimic_phewas_labels.parquet')
DEFAULT_MIMIC_JPG_DIR = "/cbica/projects/CXR/data_p/mimic-cxr-jpg/"

# Training hyper-parameters (identical to concept-based linear probing)
LR = 2e-4
WEIGHT_DECAY = 1e-8
MAX_EPOCHS = 200
PATIENCE = 10
TRAIN_BATCH_SIZE = 512

# Per-model default encoding batch sizes (tuned for GH200 80 GB)
DEFAULT_BATCH_SIZES = {
    'ark_plus': 32,
    'rad_dino': 64,
    'chexzero': 128,
    'biomedclip': 128,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_gpu_memory(stage=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        res = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"  [GPU] {stage}: {alloc:.2f} GB allocated, {res:.2f} GB reserved")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class MIMICJPEGDataset(Dataset):
    """Load MIMIC-CXR images from the JPEG directory as PIL RGB."""

    def __init__(self, df: pd.DataFrame, mimic_jpg_dir: str, transform=None):
        self.transform = transform
        self.paths: List[str] = []
        for _, row in df.iterrows():
            sid = str(row['subject_id'])
            prefix = f"p{sid[:2]}"
            path = os.path.join(
                mimic_jpg_dir, 'files', prefix,
                f"p{sid}", f"s{row['study_id']}",
                f"{row['dicom_id']}.jpg")
            self.paths.append(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class MIMICH5PILDataset(Dataset):
    """Load MIMIC-CXR images from an H5 file as PIL RGB."""

    def __init__(self, h5_path: str, num_samples: int, transform=None):
        import h5py
        self.h5_file = h5py.File(h5_path, 'r')
        self.img_dset = self.h5_file['cxr']
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.img_dset[idx]                       # float32, 0-255
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = Image.fromarray(img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def make_dataloader(df, mimic_jpg_dir, transform, batch_size,
                    h5_path=None, shuffle=False):
    if h5_path:
        dataset = MIMICH5PILDataset(h5_path, len(df), transform)
    else:
        dataset = MIMICJPEGDataset(df, mimic_jpg_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=4, pin_memory=True)


# ---------------------------------------------------------------------------
# Custom transforms
# ---------------------------------------------------------------------------
class CheXzeroTransform:
    """CheXzero preprocessing: grayscale -> 3-ch float32 [0,255] -> CXR norm."""

    def __init__(self, size=224):
        self.size = size
        self.mean = torch.tensor([101.48761] * 3).view(3, 1, 1)
        self.std = torch.tensor([83.43944] * 3).view(3, 1, 1)

    def __call__(self, img):
        img = img.convert('L')
        resample = getattr(Image, 'Resampling', Image).BICUBIC
        img = img.resize((self.size, self.size), resample)
        arr = np.array(img, dtype=np.float32)          # (H, W) 0-255
        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
        return (tensor - self.mean) / self.std


# ---------------------------------------------------------------------------
# Ark+ model (minimal re-implementation to avoid Ark repo dependency)
# ---------------------------------------------------------------------------
def _build_ark_swin(num_classes_list, projector_features, **swin_kwargs):
    """Build an Ark-style SwinTransformer with projector + omni heads."""
    from timm.models.swin_transformer import SwinTransformer

    class _ArkSwin(SwinTransformer):
        def __init__(self, ncl, pf, **kw):
            super().__init__(**kw)
            encoder_dim = self.num_features
            self.projector = nn.Linear(encoder_dim, pf)
            self.num_features = pf
            self.omni_heads = nn.ModuleList([
                nn.Linear(pf, nc) if nc > 0 else nn.Identity()
                for nc in ncl
            ])

        def generate_embeddings(self, x, after_proj=True):
            x = self.forward_features(x)
            if after_proj and self.projector is not None:
                x = self.projector(x)
            return x

    return _ArkSwin(num_classes_list, projector_features, **swin_kwargs)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_ark_plus(checkpoint_path, device):
    """Load Ark+ Swin-Large-768 with projector (1376-d)."""
    print("Loading Ark+ Swin-Large-768 ...")
    model = _build_ark_swin(
        num_classes_list=[14, 14, 14, 3, 6, 1],
        projector_features=1376,
        img_size=768, patch_size=4, window_size=12,
        embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)
    state_dict = checkpoint.get('teacher',
                  checkpoint.get('state_dict', checkpoint))
    if any('module.' in k for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v
                      for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"  Ark+ loaded (missing={len(msg.missing_keys)}, "
          f"unexpected={len(msg.unexpected_keys)})")

    model = model.to(device).eval()
    xform = transforms.Compose([
        transforms.Resize((896, 896)),
        transforms.CenterCrop(768),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return model, xform, 1376


def load_rad_dino(device):
    """Load RAD-DINO (DINOv2 ViT-B/14) from HuggingFace."""
    from transformers import AutoModel, AutoImageProcessor

    print("Loading RAD-DINO ...")
    model = AutoModel.from_pretrained("microsoft/rad-dino")
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    model = model.to(device).eval()

    xform = transforms.Compose([
        transforms.Resize(
            518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std),
    ])

    print(f"  RAD-DINO loaded  (input 518x518, feat_dim=768)")
    return model, xform, 768


def load_chexzero(checkpoint_path, device):
    """Load CheXzero (CLIP ViT-B/32) with CXR-specific normalization."""
    print("Loading CheXzero ...")
    try:
        from clip.model import CLIP as CLIPModel
    except ImportError:
        raise ImportError(
            "The 'clip' package is required for CheXzero.\n"
            "Install with: pip install git+https://github.com/openai/CLIP.git")

    model = CLIPModel(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    )
    state_dict = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=True)
    model.load_state_dict(state_dict)
    # CheXzero trains in fp16 via convert_weights(); loading into fp32 model
    # upcasts automatically.  Results are numerically equivalent for linear probing.
    model = model.to(device).eval()

    xform = CheXzeroTransform(size=224)

    print(f"  CheXzero loaded  (input 224x224, feat_dim=512)")
    return model, xform, 512


def load_biomedclip(device):
    """Load BiomedCLIP (ViT-B/16 + PubMedBERT) from HuggingFace via open_clip."""
    print("Loading BiomedCLIP ...")
    try:
        from open_clip import create_model_from_pretrained
    except ImportError:
        raise ImportError(
            "open_clip is required for BiomedCLIP.\n"
            "Install with: pip install open-clip-torch")

    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, preprocess = create_model_from_pretrained(model_name)
    model = model.to(device).eval()

    # Detect feature dimension
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        feat_dim = model.encode_image(dummy).shape[1]

    print(f"  BiomedCLIP loaded  (input 224x224, feat_dim={feat_dim})")
    return model, preprocess, feat_dim


def load_model_and_transform(model_name, args, device):
    """Dispatch to model-specific loader.  Returns (model, transform, feat_dim)."""
    if model_name == 'ark_plus':
        if not args.ark_checkpoint:
            raise ValueError("--ark_checkpoint is required for ark_plus")
        return load_ark_plus(args.ark_checkpoint, device)
    elif model_name == 'rad_dino':
        return load_rad_dino(device)
    elif model_name == 'chexzero':
        if not args.chexzero_checkpoint:
            raise ValueError("--chexzero_checkpoint is required for chexzero")
        return load_chexzero(args.chexzero_checkpoint, device)
    elif model_name == 'biomedclip':
        return load_biomedclip(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_images(model_name, model, loader, device):
    """Extract L2-normalised image features with a foundation model."""
    all_features = []
    for batch in tqdm(loader, desc=f"Encoding ({model_name})"):
        imgs = batch.to(device)

        if model_name == 'ark_plus':
            feats = model.generate_embeddings(imgs, after_proj=True)
        elif model_name == 'rad_dino':
            outputs = model(pixel_values=imgs)
            feats = outputs.pooler_output                # CLS token
        elif model_name == 'chexzero':
            feats = model.encode_image(imgs)
        elif model_name == 'biomedclip':
            feats = model.encode_image(imgs)
        else:
            raise ValueError(model_name)

        feats = F.normalize(feats.float(), dim=-1)
        all_features.append(feats.cpu())

    return torch.cat(all_features)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_splits(labels_path):
    """Load labels for all splits from parquet or CSV.

    Auto-detects phecode columns (all columns not in META_COLUMNS).
    Returns: dict of {split_name: (df, y_labels, label_cols)}
    """
    if labels_path.endswith('.parquet'):
        full_df = pd.read_parquet(labels_path)
    else:
        full_df = pd.read_csv(labels_path)

    # Auto-detect phenotype columns
    label_cols = [c for c in full_df.columns if c not in META_COLUMNS]
    print(f"  Detected {len(label_cols)} phenotype columns")

    # Safety checks
    assert full_df['dicom_id'].duplicated().sum() == 0, "Duplicate dicom_ids in labels"
    label_vals = full_df[label_cols].values
    assert label_vals.min() >= 0 and label_vals.max() <= 1, "Non-binary label values"
    assert not np.any(np.isnan(label_vals)), "NaN values in labels"

    splits = {}
    for split in ['train', 'validate', 'test']:
        df = full_df[full_df['split'] == split].reset_index(drop=True)
        y = df[label_cols].values.astype(np.float32)
        splits[split] = (df, y, label_cols)
        n_pos_cols = (y.sum(axis=0) > 0).sum()
        print(f"  {split}: {len(df)} images, {len(label_cols)} labels "
              f"({n_pos_cols} with >0 positives)")

    # Verify no patient overlap
    for s1, s2 in [('train', 'test'), ('train', 'validate'), ('validate', 'test')]:
        p1 = set(splits[s1][0]['subject_id'])
        p2 = set(splits[s2][0]['subject_id'])
        assert len(p1 & p2) == 0, f"Patient overlap between {s1} and {s2}"
    print("  No patient overlap between splits")

    return splits


# ---------------------------------------------------------------------------
# Linear probe training  (same logic as exp_linear_mimic.py)
# ---------------------------------------------------------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_epoch(model, criterion, optimizer, loader, device):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_t, all_p = [], []
    for inputs, targets in loader:
        outputs = model(inputs.to(device))
        all_t.append(targets.numpy())
        all_p.append(outputs.cpu().numpy())
    return np.concatenate(all_t), np.concatenate(all_p)


def compute_aucs(y_true, y_pred, labels, min_positives=MIN_TEST_POSITIVES):
    aucs = {}
    for i, label in enumerate(labels):
        n_pos = int(y_true[:, i].sum())
        if n_pos < min_positives or n_pos == len(y_true):
            continue
        aucs[label] = roc_auc_score(y_true[:, i], y_pred[:, i])
    return aucs


def run_single_seed(train_repr, train_labels, val_repr, val_labels,
                    test_repr, test_labels, label_cols, seed, device):
    set_random_seed(seed)

    input_dim = train_repr.shape[1]
    output_dim = len(label_cols)

    train_ds = TensorDataset(torch.from_numpy(train_repr).float(),
                             torch.from_numpy(train_labels).float())
    val_ds = TensorDataset(torch.from_numpy(val_repr).float(),
                           torch.from_numpy(val_labels).float())
    test_ds = TensorDataset(torch.from_numpy(test_repr).float(),
                            torch.from_numpy(test_labels).float())

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=TRAIN_BATCH_SIZE)

    lr_model = LogisticRegressionModel(input_dim, output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(lr_model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)

    best_val_auc = 0.0
    best_state = None
    patience_ctr = 0

    for epoch in range(MAX_EPOCHS):
        loss = train_epoch(lr_model, criterion, optimizer, train_loader,
                           device)

        y_t_val, y_p_val = evaluate_model(lr_model, val_loader, device)
        val_aucs = compute_aucs(y_t_val, y_p_val, label_cols,
                                min_positives=1)
        val_mean = np.mean(list(val_aucs.values())) if val_aucs else 0.0

        if epoch % 20 == 0 or epoch < 5:
            print(f"  Epoch {epoch + 1:3d}: loss={loss:.4f}  "
                  f"val_AUC={val_mean:.4f}")

        if val_mean > best_val_auc:
            best_val_auc = val_mean
            best_state = {k: v.cpu().clone()
                          for k, v in lr_model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    lr_model.load_state_dict(best_state)
    lr_model.to(device)

    y_true_test, y_pred_test = evaluate_model(lr_model, test_loader, device)
    test_aucs = compute_aucs(y_true_test, y_pred_test, label_cols)
    test_mean = np.mean(list(test_aucs.values())) if test_aucs else 0.0

    print(f"  Seed {seed}: val={best_val_auc:.4f}  "
          f"test_macro={test_mean:.4f} ({len(test_aucs)} phenotypes)")

    return {
        'seed': seed,
        'best_val_auc': best_val_auc,
        'test_mean_auc': test_mean,
        'per_label_aucs': test_aucs,
        'y_true_test': y_true_test,
        'y_pred_test': y_pred_test,
        'model_state': best_state,
        'model_config': {'input_dim': input_dim, 'output_dim': output_dim},
    }


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------
def save_results(model_summary, seed_results, label_cols, output_dir,
                 model_name):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir,
                            f"linear_mimic_foundation_{model_name}")
    os.makedirs(save_dir, exist_ok=True)

    summary = {
        'timestamp': ts,
        'method': 'foundation_linear_probing_mimic_phenotype',
        'foundation_model': model_name,
        **model_summary,
    }
    with open(os.path.join(save_dir, f"summary_{ts}.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    pred_dir = os.path.join(save_dir, 'predictions')
    model_dir = os.path.join(save_dir, 'models')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    for res in seed_results:
        s = res['seed']
        with open(os.path.join(pred_dir,
                               f"seed_{s}_predictions.pkl"), 'wb') as f:
            pickle.dump({
                'seed': s, 'labels': label_cols,
                'y_true': res['y_true_test'],
                'y_pred': res['y_pred_test'],
            }, f)
        torch.save({
            'seed': s,
            'state_dict': res['model_state'],
            'model_config': res['model_config'],
            'labels': label_cols,
            'test_auc': res['test_mean_auc'],
            'val_auc': res['best_val_auc'],
        }, os.path.join(model_dir, f"seed_{s}_model.pth"))

    best_res = seed_results[0]
    pd.DataFrame(
        best_res['y_pred_test'],
        columns=[f"{l}_pred" for l in label_cols]
    ).to_csv(os.path.join(save_dir, f"predictions_{ts}.csv"), index=False)
    pd.DataFrame(
        best_res['y_true_test'],
        columns=[f"{l}_true" for l in label_cols]
    ).to_csv(os.path.join(save_dir, f"ground_truth_{ts}.csv"), index=False)

    print(f"  Results saved to {save_dir}/")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_to_run = (['ark_plus', 'rad_dino', 'chexzero', 'biomedclip']
                     if args.model == 'all' else [args.model])

    print("=" * 70)
    print("Foundation Model Linear Probing on MIMIC-CXR")
    print("  PheWAS Phenotypes (full evaluation)")
    print(f"  Models: {models_to_run}")
    print(f"  Seeds:  {args.n_seeds}")
    print(f"  Device: {device}")
    print("=" * 70)

    # ---- Load split labels ----
    print("\n[Step 1] Loading MIMIC-CXR split labels ...")
    splits = load_all_splits(args.labels)
    train_df, train_labels, label_cols = splits['train']
    val_df, val_labels, _ = splits['validate']
    test_df, test_labels, _ = splits['test']

    h5_split_paths = {
        'train': args.h5_train,
        'validate': args.h5_val,
        'test': args.h5_test,
    }
    splits_data = {
        'train': (train_df, train_labels),
        'validate': (val_df, val_labels),
        'test': (test_df, test_labels),
    }

    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    seeds = list(range(42, 42 + args.n_seeds))

    all_results = {}

    for model_name in models_to_run:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name.upper()}")
        print(f"{'=' * 70}")

        # ---- Load model ----
        try:
            model, xform, feat_dim = load_model_and_transform(
                model_name, args, device)
        except (ValueError, ImportError) as e:
            print(f"  SKIP {model_name}: {e}")
            continue
        print_gpu_memory("after model load")

        # ---- Encode / cache features ----
        batch_size = args.image_batch_size or DEFAULT_BATCH_SIZES[model_name]
        image_features = {}

        for split_name, (split_df, _) in splits_data.items():
            cache_path = os.path.join(
                cache_dir, f"mimic_{split_name}_{model_name}.pt")

            if os.path.exists(cache_path) and not args.no_cache:
                print(f"  Loading cached {split_name} features: {cache_path}")
                image_features[split_name] = torch.load(
                    cache_path, weights_only=True)
            else:
                print(f"  Encoding {split_name} ({len(split_df)} images, "
                      f"bs={batch_size}) ...")
                loader = make_dataloader(
                    split_df, args.mimic_jpg_dir, xform, batch_size,
                    h5_path=h5_split_paths[split_name])
                feats = encode_images(model_name, model, loader, device)
                image_features[split_name] = feats
                torch.save(feats, cache_path)
                print(f"    Cached -> {cache_path}  "
                      f"shape={feats.shape}")

        # Free vision model
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print_gpu_memory("after freeing vision model")

        # ---- Train linear probes ----
        print(f"\n  Training linear probes ({args.n_seeds} seeds, "
              f"feat_dim={feat_dim}) ...")
        split_repr = {
            s: image_features[s].numpy() for s in ['train', 'validate', 'test']
        }
        del image_features
        gc.collect()

        seed_results = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            res = run_single_seed(
                split_repr['train'], train_labels,
                split_repr['validate'], val_labels,
                split_repr['test'], test_labels,
                label_cols, seed, device)
            seed_results.append(res)

        # ---- Aggregate ----
        test_aucs_all = [r['test_mean_auc'] for r in seed_results]
        val_aucs_all = [r['best_val_auc'] for r in seed_results]

        per_label_stats = {}
        for label in label_cols:
            la = [r['per_label_aucs'].get(label, np.nan) for r in seed_results]
            la = [a for a in la if not np.isnan(a)]
            if la:
                per_label_stats[label] = {
                    'mean': float(np.mean(la)),
                    'std': float(np.std(la)),
                }

        n_eval = len(per_label_stats)
        model_summary = {
            'foundation_model': model_name,
            'feature_dim': feat_dim,
            'n_seeds': args.n_seeds,
            'seeds': seeds,
            'n_train': len(train_labels),
            'n_val': len(val_labels),
            'n_test': len(test_labels),
            'n_phenotypes_total': len(label_cols),
            'n_phenotypes_eval': n_eval,
            'test_macro_auc': {
                'mean': float(np.mean(test_aucs_all)),
                'std': float(np.std(test_aucs_all)),
            },
            'val_auc': {
                'mean': float(np.mean(val_aucs_all)),
                'std': float(np.std(val_aucs_all)),
            },
            'per_label_stats': per_label_stats,
        }

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"Results: {model_name.upper()} (feat_dim={feat_dim}, "
              f"{args.n_seeds} seeds)")
        print(f"{'=' * 60}")
        print(f"  Macro AUROC ({n_eval} phenotypes):  "
              f"{np.mean(test_aucs_all):.4f} +/- {np.std(test_aucs_all):.4f}")
        print(f"  Val AUC:         "
              f"{np.mean(val_aucs_all):.4f} +/- {np.std(val_aucs_all):.4f}")

        # Top/bottom 5
        sorted_labels = sorted(per_label_stats.items(),
                                key=lambda x: x[1]['mean'], reverse=True)
        print(f"  Top 5:")
        for label, st in sorted_labels[:5]:
            print(f"    {label:20s} {st['mean']:.4f} +/- {st['std']:.4f}")
        print(f"  Bottom 5:")
        for label, st in sorted_labels[-5:]:
            print(f"    {label:20s} {st['mean']:.4f} +/- {st['std']:.4f}")

        save_results(model_summary, seed_results, label_cols, output_dir,
                     model_name)
        all_results[model_name] = model_summary

        del split_repr
        gc.collect()

    # ---- Final comparison ----
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Model':20s} {'Macro AUROC':>15s} {'#Pheno':>8s} {'Dim':>6s}")
        print("-" * 55)
        for mn, m in all_results.items():
            a = m['test_macro_auc']
            print(f"{mn:20s} "
                  f"{a['mean']:.4f}+/-{a['std']:.4f} "
                  f"{m['n_phenotypes_eval']:>8d} "
                  f"{m['feature_dim']:>6d}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Foundation model linear probing on MIMIC-CXR "
                    "(PheWAS phenotypes)")

    p.add_argument('--model', type=str, required=True,
                   choices=['ark_plus', 'rad_dino', 'chexzero',
                            'biomedclip', 'all'],
                   help='Foundation model to evaluate (or "all")')

    # Data paths
    p.add_argument('--labels', type=str, default=DEFAULT_LABELS)
    p.add_argument('--mimic_jpg_dir', type=str,
                   default=DEFAULT_MIMIC_JPG_DIR)
    p.add_argument('--h5_train', type=str, default=None)
    p.add_argument('--h5_val', type=str, default=None)
    p.add_argument('--h5_test', type=str, default=None)

    # Model checkpoints
    p.add_argument('--ark_checkpoint', type=str, default=None,
                   help='Path to Ark+ .pth.tar checkpoint')
    p.add_argument('--chexzero_checkpoint', type=str, default=None,
                   help='Path to CheXzero .pt checkpoint')

    # Experiment settings
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--image_batch_size', type=int, default=None,
                   help='Override per-model default batch size')
    p.add_argument('--no_cache', action='store_true',
                   help='Re-encode images even if cache exists')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
