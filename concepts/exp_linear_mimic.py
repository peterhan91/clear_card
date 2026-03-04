#!/usr/bin/env python3
"""
Supervised concept-based linear probing on MIMIC-CXR for 27 opportunistic
screening phenotypes (CXR-Phenotype project).

Pipeline:
  1. Load finetuned ViT-7B LoRA CLIP model (DINOv3 backbone), merge LoRA
  2. Load ~492k radiological concepts and their precomputed LLM embeddings
  3. Encode concepts via CLIP text encoder -> concept_features  [N_concepts, 768]
  4. Encode MIMIC-CXR train/val/test images via CLIP vision encoder
  5. Concept similarity: scores = image_features @ concept_features.T
  6. Project to LLM space: llm_repr = scores @ concept_embeddings  [N_images, emb_dim]
  7. Train logistic regression on train split, early-stop on val split
  8. Evaluate AUC on test split with bootstrap CIs
  9. Repeat for multiple random seeds

Usage:
  python exp_linear_mimic.py [--embedding_models openai_3large sfr_mistral]
                              [--model_path /path/to/best_model.pt]
                              [--labels_csv data/mimic_opportunistic_labels.csv]
                              [--mimic_jpg_dir /path/to/mimic-cxr-jpg/]
                              [--n_seeds 5]
"""
import os
import sys
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
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score
import h5py

# Add training directory for model loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from zero_shot import load_clip
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

# Labels with 0 test positives — cannot compute AUC
SKIP_EVAL_LABELS = {'pulmonary_valve'}

# Core phenotypes: cardiac + pulmonary, most CXR-relevant
CORE_PHENOTYPES = [
    'heart_failure', 'hfref', 'hfpef', 'atrial_fibrillation',
    'cad', 'pulmonary_htn', 'copd', 'lung_cancer',
]

# Best ViT-7B LoRA model (combined CheXpert+ReXGradient)
DEFAULT_MODEL_PATH = (
    "/cbica/projects/CXR/codes/clear_card/checkpoints/"
    "dinov3-vit7b16-lora-r32-a64-combined_vit7b16_20260224_144733/best_model.pt"
)

# Data paths
DEFAULT_LABELS_CSV = os.path.join(os.path.dirname(__file__), '..', 'data',
                                   'mimic_opportunistic_labels.csv')
DEFAULT_MIMIC_JPG_DIR = "/cbica/projects/CXR/data_p/mimic-cxr-jpg/"

# Concepts
CONCEPTS_CSV = os.path.join(os.path.dirname(__file__), 'concepts_clean.csv')
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), 'embeddings_output')

# DINOv3
DINOV3_REPO = "/cbica/projects/CXR/codes/dinov3"
DINOV3_WEIGHTS = "/cbica/projects/CXR/codes/dinov3/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"

# Embedding model mapping
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
    'openai_3large': {
        'pickle_suffix': 'openai_3large_3072d',
        'dim': 3072,
    },
}

# Training hyperparameters
LR = 2e-4
WEIGHT_DECAY = 1e-8
MAX_EPOCHS = 200
PATIENCE = 10
TRAIN_BATCH_SIZE = 512


def print_gpu_memory(stage=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        res = torch.cuda.memory_reserved() / 1024**3
        print(f"  [GPU] {stage}: {alloc:.2f} GB allocated, {res:.2f} GB reserved")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def load_clip_model(model_path: str, merge_lora: bool = True):
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
        lora_dropout=0.0,
    )

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
# Datasets
# ---------------------------------------------------------------------------
class MIMICCXRDataset(Dataset):
    """Load MIMIC-CXR images from JPEG directory structure."""

    def __init__(self, df: pd.DataFrame, mimic_jpg_dir: str, transform=None):
        self.mimic_jpg_dir = mimic_jpg_dir
        self.transform = transform

        self.paths = []
        for _, row in df.iterrows():
            sid = str(row['subject_id'])
            prefix = f"p{sid[:2]}"
            path = os.path.join(mimic_jpg_dir, 'files', prefix,
                                f"p{sid}", f"s{row['study_id']}",
                                f"{row['dicom_id']}.jpg")
            self.paths.append(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        return img


class MIMICH5Dataset(Dataset):
    """Load MIMIC-CXR images from a pre-converted H5 file."""

    def __init__(self, h5_path: str, num_samples: int, transform=None):
        self.h5_file = h5py.File(h5_path, 'r')
        self.img_dset = self.h5_file['cxr']
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.img_dset[idx]
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        return img

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_splits(labels_csv: str):
    """Load labels for all splits from a single CSV read.

    Returns: dict of {split_name: (df, y_labels, label_cols)}
    """
    full_df = pd.read_csv(labels_csv)
    label_cols = [c for c in PHENOTYPE_LABELS if c in full_df.columns]

    splits = {}
    for split in ['train', 'validate', 'test']:
        df = full_df[full_df['split'] == split].reset_index(drop=True)
        y = df[label_cols].values.astype(float)
        y = np.nan_to_num(y, nan=0.0)
        y[y < 0] = 0.0
        splits[split] = (df, y, label_cols)
        print(f"  {split}: {len(df)} images, {len(label_cols)} labels")

    return splits


def make_dataloader(df: pd.DataFrame, mimic_jpg_dir: str,
                    batch_size: int, h5_path: Optional[str] = None,
                    shuffle: bool = False):
    """Create a DataLoader for a split.

    Args:
        h5_path: Optional per-split H5 file. Must contain exactly
                 len(df) images in the same order as df.
    """
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761),
                  (83.43944, 83.43944, 83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])

    if h5_path:
        dataset = MIMICH5Dataset(h5_path=h5_path,
                                 num_samples=len(df),
                                 transform=transform)
    else:
        dataset = MIMICCXRDataset(df=df, mimic_jpg_dir=mimic_jpg_dir,
                                   transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=2, pin_memory=True)
    return loader


# ---------------------------------------------------------------------------
# Concept loading
# ---------------------------------------------------------------------------
def load_concepts(max_concepts: int = 0):
    df = pd.read_csv(CONCEPTS_CSV)
    concepts = df['concept'].tolist()
    indices = df['concept_idx'].tolist()
    if max_concepts > 0:
        concepts = concepts[:max_concepts]
        indices = indices[:max_concepts]
    print(f"Loaded {len(concepts)} concepts")
    return concepts, indices


def load_concept_embeddings(model_key: str, concept_indices: List[int]):
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

    return torch.from_numpy(embeddings)


# ---------------------------------------------------------------------------
# CLIP feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_concepts_clip(model, concepts: List[str], batch_size: int = 512):
    """Encode concepts through CLIP text encoder."""
    all_features = []
    for i in tqdm(range(0, len(concepts), batch_size), desc="Encoding concepts"):
        batch = concepts[i:i + batch_size]
        tokens = clip_module.tokenize(batch, context_length=77).cuda()
        feats = model.encode_text(tokens)
        feats = F.normalize(feats.float(), dim=-1)
        all_features.append(feats.cpu())
        torch.cuda.empty_cache()
    return torch.cat(all_features)


@torch.no_grad()
def encode_images_clip(model, loader):
    """Encode images through CLIP vision encoder."""
    all_features = []
    for batch in tqdm(loader, desc="Encoding images"):
        imgs = batch.cuda()
        feats = model.encode_image(imgs)
        feats = F.normalize(feats.float(), dim=-1)
        all_features.append(feats.cpu())
    return torch.cat(all_features)


# ---------------------------------------------------------------------------
# LLM-space projection
# ---------------------------------------------------------------------------
@torch.no_grad()
def project_to_llm_space(image_features: torch.Tensor,
                          concept_features: torch.Tensor,
                          concept_embeddings: torch.Tensor,
                          batch_size: int = 500):
    """
    Compute concept similarities and project to LLM embedding space.

    image_features: [N_images, 768]
    concept_features: [N_concepts, 768]
    concept_embeddings: [N_concepts, emb_dim]

    Returns: [N_images, emb_dim] normalized LLM representations on CPU.
    """
    ce = concept_embeddings.cuda().float()
    cf = concept_features.cuda().float()

    all_repr = []
    for i in tqdm(range(0, len(image_features), batch_size),
                  desc="Projecting to LLM space"):
        img_batch = image_features[i:i + batch_size].cuda().float()

        # Concept similarity: [batch, N_concepts]
        sim = img_batch @ cf.T

        # Project to LLM space: [batch, emb_dim]
        llm_batch = sim @ ce
        llm_batch = torch.clamp(llm_batch, min=-1e6, max=1e6)
        norms = llm_batch.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        llm_batch = llm_batch / norms

        all_repr.append(llm_batch.cpu())
        torch.cuda.empty_cache()

    return torch.cat(all_repr)


# ---------------------------------------------------------------------------
# Linear probing training
# ---------------------------------------------------------------------------
def train_epoch(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()
    all_targets = []
    all_preds = []
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_targets.append(targets.numpy())
        all_preds.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return y_true, y_pred


def compute_aucs(y_true, y_pred, labels, skip_labels=None):
    """Compute per-label AUCs, skipping labels with 0 positives."""
    skip_labels = skip_labels or set()
    aucs = {}
    for i, label in enumerate(labels):
        if label in skip_labels:
            continue
        n_pos = int(y_true[:, i].sum())
        if n_pos == 0 or n_pos == len(y_true):
            continue
        aucs[label] = roc_auc_score(y_true[:, i], y_pred[:, i])
    return aucs


def run_single_seed(train_repr, train_labels, val_repr, val_labels,
                    test_repr, test_labels, label_cols, seed, device):
    """Train and evaluate one seed of linear probing."""
    set_random_seed(seed)

    input_dim = train_repr.shape[1]
    output_dim = len(label_cols)

    # Data loaders
    train_ds = TensorDataset(torch.from_numpy(train_repr).float(),
                             torch.from_numpy(train_labels).float())
    val_ds = TensorDataset(torch.from_numpy(val_repr).float(),
                           torch.from_numpy(val_labels).float())
    test_ds = TensorDataset(torch.from_numpy(test_repr).float(),
                            torch.from_numpy(test_labels).float())

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=TRAIN_BATCH_SIZE,
                             shuffle=False)

    # Model
    lr_model = LogisticRegressionModel(input_dim, output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(lr_model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)

    # Training loop with early stopping
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(lr_model, criterion, optimizer, train_loader,
                                 device)

        # Validation
        y_true_val, y_pred_val = evaluate_model(lr_model, val_loader, device)
        val_aucs = compute_aucs(y_true_val, y_pred_val, label_cols,
                                SKIP_EVAL_LABELS)
        val_mean_auc = np.mean(list(val_aucs.values())) if val_aucs else 0.0

        if epoch % 20 == 0 or epoch < 5:
            print(f"  Epoch {epoch + 1:3d}: loss={train_loss:.4f}  "
                  f"val_AUC={val_mean_auc:.4f}")

        if val_mean_auc > best_val_auc:
            best_val_auc = val_mean_auc
            best_model_state = {k: v.cpu().clone()
                                for k, v in lr_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    lr_model.load_state_dict(best_model_state)
    lr_model.to(device)

    # Evaluate on test set
    y_true_test, y_pred_test = evaluate_model(lr_model, test_loader, device)
    test_aucs = compute_aucs(y_true_test, y_pred_test, label_cols,
                             SKIP_EVAL_LABELS)
    test_mean_auc = np.mean(list(test_aucs.values())) if test_aucs else 0.0

    core_aucs = [test_aucs[c] for c in CORE_PHENOTYPES if c in test_aucs]
    core_mean_auc = np.mean(core_aucs) if core_aucs else 0.0

    print(f"  Seed {seed}: val_AUC={best_val_auc:.4f}  "
          f"test_AUC={test_mean_auc:.4f}  "
          f"core_AUC={core_mean_auc:.4f}")

    return {
        'seed': seed,
        'best_val_auc': best_val_auc,
        'test_mean_auc': test_mean_auc,
        'test_core_auc': core_mean_auc,
        'per_label_aucs': test_aucs,
        'y_true_test': y_true_test,
        'y_pred_test': y_pred_test,
        'model_state': best_model_state,
        'model_config': {
            'input_dim': input_dim,
            'output_dim': output_dim,
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_linear_probing_pipeline(args):
    """Run the full concept-based supervised linear probing pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Concept-Based Supervised Linear Probing on MIMIC-CXR")
    print("  27 Opportunistic Screening Phenotypes (CXR-Phenotype)")
    print("  Model: DINOv3 ViT-7B + LoRA (r=32, alpha=64)")
    print(f"  Embedding models: {args.embedding_models}")
    print(f"  Seeds: {args.n_seeds}")
    print(f"  Labels CSV: {args.labels_csv}")
    print(f"  Image source: JPEG ({args.mimic_jpg_dir})")
    if args.h5_train or args.h5_val or args.h5_test:
        print(f"  H5 overrides: train={args.h5_train}, "
              f"val={args.h5_val}, test={args.h5_test}")
    print("=" * 70)

    # Step 1: Load split metadata
    print("\n[Step 1/7] Loading MIMIC-CXR split labels...")
    splits = load_all_splits(args.labels_csv)
    train_df, train_labels, label_cols = splits['train']
    val_df, val_labels, _ = splits['validate']
    test_df, test_labels, _ = splits['test']

    # Step 2: Load CLIP model
    print("\n[Step 2/7] Loading ViT-7B LoRA CLIP model...")
    clip_model = load_clip_model(args.model_path, merge_lora=args.merge_lora)

    # Step 3: Load concepts
    print("\n[Step 3/7] Loading concepts...")
    concepts, concept_indices = load_concepts(
        max_concepts=args.max_concepts)

    # Step 4: Encode concepts with CLIP text encoder
    print("\n[Step 4/7] Encoding concepts with CLIP text encoder...")
    concept_features = encode_concepts_clip(clip_model, concepts,
                                             batch_size=args.concept_batch_size)
    print(f"  Concept features: {concept_features.shape}")

    # Step 5: Encode images for all splits
    print("\n[Step 5/7] Encoding images for all splits...")
    feature_cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(feature_cache_dir, exist_ok=True)

    # Map split names to per-split H5 paths (if provided)
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
    image_features = {}

    for split_name, (split_df, _) in splits_data.items():
        cache_path = os.path.join(feature_cache_dir,
                                  f"mimic_{split_name}_image_features.pt")
        if os.path.exists(cache_path) and not args.no_cache:
            print(f"  Loading cached {split_name} features from {cache_path}")
            image_features[split_name] = torch.load(cache_path,
                                                     weights_only=True)
        else:
            print(f"  Encoding {split_name} images ({len(split_df)} images)...")
            loader = make_dataloader(
                split_df, args.mimic_jpg_dir,
                batch_size=args.image_batch_size,
                h5_path=h5_split_paths[split_name],
                shuffle=False,
            )
            feats = encode_images_clip(clip_model, loader)
            image_features[split_name] = feats
            torch.save(feats, cache_path)
            print(f"  Cached to {cache_path}")

    # Free CLIP model
    print("\n  Freeing CLIP model from GPU...")
    del clip_model
    torch.cuda.empty_cache()
    gc.collect()
    print_gpu_memory("after freeing CLIP")

    # Step 6-7: For each embedding model, project & train linear probe
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    all_model_results = {}
    seeds = list(range(42, 42 + args.n_seeds))

    for model_key in args.embedding_models:
        print(f"\n{'=' * 70}")
        print(f"[Step 6/7] Embedding model: {model_key.upper()}")
        print(f"{'=' * 70}")

        # Load precomputed concept embeddings
        concept_embeds = load_concept_embeddings(model_key, concept_indices)

        # Project all splits to LLM space
        split_repr = {}
        for split_name in ['train', 'validate', 'test']:
            print(f"  Projecting {split_name} to LLM space...")
            repr_tensor = project_to_llm_space(
                image_features[split_name], concept_features, concept_embeds,
                batch_size=500,
            )
            split_repr[split_name] = repr_tensor.numpy()
            print(f"    {split_name} LLM repr: {split_repr[split_name].shape}")

        del concept_embeds
        torch.cuda.empty_cache()
        gc.collect()

        # Run multi-seed linear probing
        print(f"\n[Step 7/7] Training linear probes ({args.n_seeds} seeds)...")
        seed_results = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            res = run_single_seed(
                split_repr['train'], train_labels,
                split_repr['validate'], val_labels,
                split_repr['test'], test_labels,
                label_cols, seed, device,
            )
            seed_results.append(res)

        # Aggregate results
        test_aucs_all = [r['test_mean_auc'] for r in seed_results]
        core_aucs_all = [r['test_core_auc'] for r in seed_results]
        val_aucs_all = [r['best_val_auc'] for r in seed_results]

        per_label_stats = {}
        for label in label_cols:
            if label in SKIP_EVAL_LABELS:
                continue
            label_aucs = [r['per_label_aucs'].get(label, np.nan)
                          for r in seed_results]
            label_aucs = [a for a in label_aucs if not np.isnan(a)]
            if label_aucs:
                per_label_stats[label] = {
                    'mean': float(np.mean(label_aucs)),
                    'std': float(np.std(label_aucs)),
                }

        model_summary = {
            'embedding_model': model_key,
            'n_seeds': args.n_seeds,
            'seeds': seeds,
            'n_concepts': len(concepts),
            'n_train': len(train_labels),
            'n_val': len(val_labels),
            'n_test': len(test_labels),
            'test_mean_auc': {
                'mean': float(np.mean(test_aucs_all)),
                'std': float(np.std(test_aucs_all)),
            },
            'test_core_auc': {
                'mean': float(np.mean(core_aucs_all)),
                'std': float(np.std(core_aucs_all)),
            },
            'val_auc': {
                'mean': float(np.mean(val_aucs_all)),
                'std': float(np.std(val_aucs_all)),
            },
            'per_label_stats': per_label_stats,
        }

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"Results for {model_key.upper()} ({len(concepts)} concepts, "
              f"{args.n_seeds} seeds)")
        print(f"{'=' * 60}")
        print(f"  Test AUC (all):  "
              f"{np.mean(test_aucs_all):.4f} +/- {np.std(test_aucs_all):.4f}")
        print(f"  Test AUC (core): "
              f"{np.mean(core_aucs_all):.4f} +/- {np.std(core_aucs_all):.4f}")
        print(f"  Val AUC:         "
              f"{np.mean(val_aucs_all):.4f} +/- {np.std(val_aucs_all):.4f}")
        print(f"  Per-label AUCs (mean +/- std):")
        for label, stats in per_label_stats.items():
            marker = " *" if label in CORE_PHENOTYPES else ""
            print(f"    {label:45s} "
                  f"{stats['mean']:.4f} +/- {stats['std']:.4f}{marker}")

        # Save results
        save_results(model_summary, seed_results, label_cols, output_dir,
                     model_key, args.model_path)

        all_model_results[model_key] = model_summary

        # Cleanup
        del split_repr
        gc.collect()

    # Final comparison
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':25s} {'Core AUC':>15s} {'All AUC':>15s}")
    print("-" * 60)
    for model_key, m in all_model_results.items():
        core = m['test_core_auc']
        all_ = m['test_mean_auc']
        print(f"{model_key:25s} "
              f"{core['mean']:.4f}+/-{core['std']:.4f} "
              f"{all_['mean']:.4f}+/-{all_['std']:.4f}")

    return all_model_results


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------
def save_results(model_summary, seed_results, label_cols, output_dir,
                 model_key, model_path):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir,
                            f"linear_mimic_{model_key}")
    os.makedirs(save_dir, exist_ok=True)

    # Summary JSON
    summary = {
        'timestamp': ts,
        'method': 'concept_linear_probing_mimic_phenotype',
        'model_path': model_path,
        'normalization': {'mean': [101.48761] * 3, 'std': [83.43944] * 3},
        'resolution': 448,
        **model_summary,
    }
    with open(os.path.join(save_dir, f"summary_{ts}.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # Per-seed predictions & models
    predictions_dir = os.path.join(save_dir, 'predictions')
    models_dir = os.path.join(save_dir, 'models')
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    for res in seed_results:
        seed = res['seed']

        # Predictions
        pred_data = {
            'seed': seed,
            'labels': label_cols,
            'y_true': res['y_true_test'],
            'y_pred': res['y_pred_test'],
        }
        with open(os.path.join(predictions_dir,
                               f"seed_{seed}_predictions.pkl"), 'wb') as f:
            pickle.dump(pred_data, f)

        # Model checkpoint
        torch.save({
            'seed': seed,
            'state_dict': res['model_state'],
            'model_config': res['model_config'],
            'labels': label_cols,
            'test_auc': res['test_mean_auc'],
            'val_auc': res['best_val_auc'],
        }, os.path.join(models_dir, f"seed_{seed}_model.pth"))

    # Save predictions CSV for the first seed (for quick inspection)
    best_res = seed_results[0]
    pd.DataFrame(best_res['y_pred_test'],
                 columns=[f"{l}_pred" for l in label_cols]).to_csv(
        os.path.join(save_dir, f"predictions_{ts}.csv"), index=False)
    pd.DataFrame(best_res['y_true_test'],
                 columns=[f"{l}_true" for l in label_cols]).to_csv(
        os.path.join(save_dir, f"ground_truth_{ts}.csv"), index=False)

    print(f"  Results saved to {save_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised concept-based linear probing on MIMIC-CXR "
                    "(27 opportunistic screening phenotypes)")

    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to best_model.pt checkpoint')
    parser.add_argument('--labels_csv', type=str, default=DEFAULT_LABELS_CSV,
                        help='Path to MIMIC opportunistic labels CSV')
    parser.add_argument('--mimic_jpg_dir', type=str,
                        default=DEFAULT_MIMIC_JPG_DIR,
                        help='Path to MIMIC-CXR JPEG directory')
    parser.add_argument('--h5_train', type=str, default=None,
                        help='Optional per-split H5 for train images')
    parser.add_argument('--h5_val', type=str, default=None,
                        help='Optional per-split H5 for validate images')
    parser.add_argument('--h5_test', type=str, default=None,
                        help='Optional per-split H5 for test images')
    parser.add_argument('--embedding_models', nargs='+',
                        default=['openai_3large'],
                        choices=list(EMBEDDING_MODELS.keys()),
                        help='Embedding models to evaluate')
    parser.add_argument('--max_concepts', type=int, default=0,
                        help='Max concepts to use (0 = all ~492k)')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Number of random seeds (default 5)')
    parser.add_argument('--image_batch_size', type=int, default=16,
                        help='Batch size for image encoding')
    parser.add_argument('--concept_batch_size', type=int, default=512,
                        help='Batch size for concept text encoding')
    parser.add_argument('--merge_lora', action='store_true', default=True)
    parser.add_argument('--no_merge_lora', dest='merge_lora',
                        action='store_false')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Disable feature caching')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_linear_probing_pipeline(args)
