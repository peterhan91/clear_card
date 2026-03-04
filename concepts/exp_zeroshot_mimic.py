#!/usr/bin/env python3
"""
Zero-shot concept-based classification on MIMIC-CXR test set for 27 opportunistic
screening phenotypes (CXR-Phenotype project).

Pipeline:
  1. Load finetuned ViT-7B LoRA CLIP model (DINOv3 backbone)
  2. Merge LoRA weights into base model for efficient inference
  3. Load ~492k radiological concepts and their precomputed LLM embeddings
  4. Encode concepts via CLIP text encoder -> concept_features  [N_concepts, 768]
  5. Encode MIMIC-CXR test images via CLIP vision encoder -> image_features  [N_images, 768]
  6. Concept similarity: scores = image_features @ concept_features.T  [N_images, N_concepts]
  7. Project to LLM space: llm_repr = scores @ concept_embeddings  [N_images, emb_dim]
  8. Embed pos/neg class prompts in same LLM space  [N_labels, emb_dim] x 2
  9. Predict: P(label) = sigmoid(cos_sim(llm_repr, pos) - cos_sim(llm_repr, neg))
 10. Evaluate AUC on 27 phenotype labels (exclude pulmonary_valve — 0 test positives)

Usage:
  python exp_zeroshot_mimic.py [--embedding_models openai_3large sfr_mistral]
                               [--model_path /path/to/best_model.pt]
                               [--labels_csv data/mimic_opportunistic_labels.csv]
                               [--mimic_jpg_dir /path/to/mimic-cxr-jpg/]
                               [--h5_path /path/to/mimic_test.h5]
"""
import os
import sys
import json
import pickle
import datetime
import argparse
import gc
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from tqdm import tqdm
from PIL import Image
import h5py

# Add training directory for model loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from zero_shot import load_clip
from eval import evaluate, bootstrap, compute_cis
import clip as clip_module  # CLIP tokenizer

# Add current directory for embedding utilities
from get_embed import RadiologyEmbeddingGenerator

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

# Prompt mapping: snake_case -> readable medical terms
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
        'prompt_mode': 'local',
    },
    'nemotron_8b': {
        'hf_name': 'nvidia/llama-embed-nemotron-8b',
        'pickle_suffix': 'nemotron_8b',
        'dim': 4096,
        'prompt_mode': 'local',
    },
    'kalm_gemma3_12b': {
        'hf_name': 'tencent/KaLM-Embedding-Gemma3-12B-2511',
        'pickle_suffix': 'kalm_gemma3_12b',
        'dim': 3840,
        'prompt_mode': 'local',
    },
    'openai_3large': {
        'pickle_suffix': 'openai_3large_3072d',
        'dim': 3072,
        'prompt_mode': 'openai_api',
    },
}


def load_env():
    """Load API keys from .env file."""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip())


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
# Dataset: MIMIC-CXR JPEG loading
# ---------------------------------------------------------------------------
class MIMICCXRDataset(Dataset):
    """Load MIMIC-CXR test images from JPEG directory structure."""

    def __init__(self, labels_csv: str, mimic_jpg_dir: str,
                 split: str = 'test', transform=None):
        self.mimic_jpg_dir = mimic_jpg_dir
        self.transform = transform

        df = pd.read_csv(labels_csv)
        df = df[df['split'] == split].reset_index(drop=True)
        self.df = df
        self.dicom_ids = df['dicom_id'].tolist()
        self.subject_ids = df['subject_id'].tolist()
        self.study_ids = df['study_id'].tolist()

        # Build file paths: files/pXX/p{subject_id}/s{study_id}/{dicom_id}.jpg
        self.paths = []
        for _, row in df.iterrows():
            sid = str(row['subject_id'])
            prefix = f"p{sid[:2]}"
            path = os.path.join(mimic_jpg_dir, 'files', prefix,
                                f"p{sid}", f"s{row['study_id']}",
                                f"{row['dicom_id']}.jpg")
            self.paths.append(path)

        print(f"MIMICCXRDataset: {len(self.paths)} images for split='{split}'")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')  # grayscale
        img = np.array(img, dtype=np.float32)
        # Expand to (1, H, W) then repeat to (3, H, W)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)

        if self.transform:
            img = self.transform(img)

        return {'img': img}


class MIMICH5Dataset(Dataset):
    """Load MIMIC-CXR test images from a pre-converted H5 file."""

    def __init__(self, h5_path: str, transform=None):
        self.img_dset = h5py.File(h5_path, 'r')['cxr']
        self.transform = transform
        print(f"MIMICH5Dataset: {len(self.img_dset)} images from {h5_path}")

    def __len__(self):
        return len(self.img_dset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.img_dset[idx]  # (H, W) float32
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        return {'img': img}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_test_data(labels_csv: str, mimic_jpg_dir: str,
                   h5_path: str = None, batch_size: int = 32):
    """Load MIMIC-CXR test dataset and ground-truth labels."""
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761),
                  (83.43944,  83.43944,  83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])

    if h5_path:
        dataset = MIMICH5Dataset(h5_path=h5_path, transform=transform)
    else:
        dataset = MIMICCXRDataset(labels_csv=labels_csv,
                                  mimic_jpg_dir=mimic_jpg_dir,
                                  split='test', transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    # Ground-truth labels
    df = pd.read_csv(labels_csv)
    df = df[df['split'] == 'test'].reset_index(drop=True)
    label_cols = [c for c in PHENOTYPE_LABELS if c in df.columns]
    y_true = df[label_cols].values.astype(float)
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_true[y_true < 0] = 0.0

    print(f"Test set: {len(dataset)} images, {len(label_cols)} labels")
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
    """Project images into LLM embedding space via concept scores."""
    batch_size = 64
    llm_repr_chunks = []
    ce = concept_embeddings.cuda().float()
    for i in range(0, len(concept_scores), batch_size):
        cs_batch = concept_scores[i:i + batch_size].cuda().float()
        llm_batch = cs_batch @ ce
        llm_batch = torch.clamp(llm_batch, min=-1e6, max=1e6)
        norms = llm_batch.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        llm_batch = llm_batch / norms
        llm_repr_chunks.append(llm_batch.cpu())
        torch.cuda.empty_cache()

    llm_repr = torch.cat(llm_repr_chunks)
    return llm_repr  # [N_images, emb_dim]


def get_prompt_texts(labels: List[str]):
    """Generate positive and negative prompt texts for each label."""
    pos_prompts = []
    neg_prompts = []
    for label in labels:
        readable = LABEL_TO_PROMPT.get(label, label.replace('_', ' '))
        pos_prompts.append(readable)
        neg_prompts.append(f"no {readable}")
    return pos_prompts, neg_prompts


def generate_prompt_embeddings_local(model_key: str, labels: List[str]):
    """Generate LLM embeddings for pos/neg prompts using a local HuggingFace model."""
    info = EMBEDDING_MODELS[model_key]
    hf_name = info['hf_name']

    pos_prompts, neg_prompts = get_prompt_texts(labels)
    all_prompts = pos_prompts + neg_prompts

    print(f"  Generating {len(all_prompts)} prompt embeddings with {hf_name}")

    generator = RadiologyEmbeddingGenerator(
        embedding_type="sentence_transformers",
        local_model_name=hf_name,
        batch_size=len(all_prompts),
        use_fp16=True,
        trust_remote_code=True,
    )

    embeddings = generator.get_sentence_transformer_embeddings_batch(all_prompts)
    embeddings = np.array(embeddings, dtype=np.float32)

    del generator
    torch.cuda.empty_cache()
    gc.collect()

    n = len(labels)
    pos_embeds = torch.from_numpy(embeddings[:n])
    neg_embeds = torch.from_numpy(embeddings[n:])

    pos_embeds = F.normalize(pos_embeds.float(), dim=-1)
    neg_embeds = F.normalize(neg_embeds.float(), dim=-1)

    return pos_embeds, neg_embeds


def generate_prompt_embeddings_openai(labels: List[str]):
    """Generate embeddings for pos/neg prompts using OpenAI text-embedding-3-large."""
    from openai import OpenAI

    load_env()
    client = OpenAI()

    pos_prompts, neg_prompts = get_prompt_texts(labels)
    all_prompts = pos_prompts + neg_prompts

    print(f"  Generating {len(all_prompts)} prompt embeddings with OpenAI text-embedding-3-large")

    response = client.embeddings.create(
        model='text-embedding-3-large',
        input=all_prompts,
    )

    embeddings = np.array(
        [item.embedding for item in response.data],
        dtype=np.float32,
    )

    n = len(labels)
    pos_embeds = torch.from_numpy(embeddings[:n])
    neg_embeds = torch.from_numpy(embeddings[n:])

    pos_embeds = F.normalize(pos_embeds.float(), dim=-1)
    neg_embeds = F.normalize(neg_embeds.float(), dim=-1)

    return pos_embeds, neg_embeds


def generate_prompt_embeddings(model_key: str, labels: List[str]):
    """Dispatch to local or OpenAI prompt embedding generation."""
    info = EMBEDDING_MODELS[model_key]
    if info.get('prompt_mode') == 'openai_api':
        return generate_prompt_embeddings_openai(labels)
    else:
        return generate_prompt_embeddings_local(model_key, labels)


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
    # Identify evaluable labels (skip those with 0 positives)
    eval_mask = []
    eval_labels = []
    for i, label in enumerate(labels):
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

    # Core-phenotype mean AUC
    core_aucs = [aucs[c] for c in CORE_PHENOTYPES if c in aucs]
    mean_core_auc = np.mean(core_aucs) if core_aucs else 0.0
    mean_all_auc = np.mean(list(aucs.values())) if aucs else 0.0

    print(f"\n{'='*60}")
    print(f"Results for {model_key.upper()} ({n_concepts} concepts)")
    print(f"{'='*60}")
    print(f"  Mean AUC ({len(core_aucs)} core):  {mean_core_auc:.4f}")
    print(f"  Mean AUC ({len(eval_labels)} eval):  {mean_all_auc:.4f}")
    print(f"  Per-label AUCs:")
    for label, auc_val in aucs.items():
        marker = " *" if label in CORE_PHENOTYPES else ""
        print(f"    {label:45s} {auc_val:.4f}{marker}")

    # Bootstrap CIs
    cis_df = None
    if run_bootstrap:
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

    return {
        'model': model_key,
        'n_concepts': n_concepts,
        'mean_core_auc': mean_core_auc,
        'mean_all_auc': mean_all_auc,
        'per_label_aucs': aucs,
        'eval_labels': eval_labels,
        'results_df': results_df,
        'cis_df': cis_df,
    }


def save_results(metrics: dict, y_pred: np.ndarray, y_true: np.ndarray,
                 labels: List[str], output_dir: str, model_path: str):
    """Save evaluation results to disk."""
    model_key = metrics['model']
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"zeroshot_mimic_{model_key}")
    os.makedirs(save_dir, exist_ok=True)

    summary = {
        'timestamp': ts,
        'method': 'concept_zeroshot_mimic_phenotype',
        'embedding_model': model_key,
        'n_concepts': metrics['n_concepts'],
        'n_test_images': len(y_pred),
        'n_labels': len(labels),
        'n_eval_labels': len(metrics['eval_labels']),
        'labels': labels,
        'eval_labels': metrics['eval_labels'],
        'mean_core_auc': metrics['mean_core_auc'],
        'mean_all_auc': metrics['mean_all_auc'],
        'per_label_aucs': metrics['per_label_aucs'],
        'model_path': model_path,
        'normalization': {'mean': [101.48761]*3, 'std': [83.43944]*3},
        'resolution': 448,
    }
    with open(os.path.join(save_dir, f"summary_{ts}.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(y_pred, columns=[f"{l}_pred" for l in labels]).to_csv(
        os.path.join(save_dir, f"predictions_{ts}.csv"), index=False)
    pd.DataFrame(y_true, columns=[f"{l}_true" for l in labels]).to_csv(
        os.path.join(save_dir, f"ground_truth_{ts}.csv"), index=False)

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
    """Run the full concept-based zero-shot classification pipeline on MIMIC-CXR."""

    print("=" * 70)
    print("Concept-Based Zero-Shot Classification on MIMIC-CXR Test Set")
    print("  27 Opportunistic Screening Phenotypes (CXR-Phenotype)")
    print("  Model: DINOv3 ViT-7B + LoRA (r=32, alpha=64)")
    print(f"  Embedding models: {args.embedding_models}")
    print(f"  Labels CSV: {args.labels_csv}")
    if args.h5_path:
        print(f"  Image source: H5 ({args.h5_path})")
    else:
        print(f"  Image source: JPEG ({args.mimic_jpg_dir})")
    print("=" * 70)

    # Step 1: Load CLIP model
    print("\n[Step 1/6] Loading ViT-7B LoRA CLIP model...")
    model = load_model(args.model_path, merge_lora=args.merge_lora)

    # Step 2: Load test data
    print("\n[Step 2/6] Loading MIMIC-CXR test data...")
    loader, y_true, label_cols = load_test_data(
        labels_csv=args.labels_csv,
        mimic_jpg_dir=args.mimic_jpg_dir,
        h5_path=args.h5_path,
        batch_size=args.image_batch_size,
    )

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

    # Free CLIP model
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

        # Generate prompt embeddings
        print("  Generating class prompt embeddings...")
        pos_embeds, neg_embeds = generate_prompt_embeddings(model_key, label_cols)
        print(f"  Pos embeddings: {pos_embeds.shape}, Neg: {neg_embeds.shape}")

        # Print prompt mapping for verification
        pos_prompts, neg_prompts = get_prompt_texts(label_cols)
        print(f"  Sample prompts: pos='{pos_prompts[0]}', neg='{neg_prompts[0]}'")

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
        save_results(metrics, y_pred, y_true, label_cols, output_dir,
                     args.model_path)

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
        description="Zero-shot concept-based classification on MIMIC-CXR test set "
                    "(27 opportunistic screening phenotypes)")

    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to best_model.pt checkpoint')
    parser.add_argument('--labels_csv', type=str, default=DEFAULT_LABELS_CSV,
                        help='Path to MIMIC opportunistic labels CSV')
    parser.add_argument('--mimic_jpg_dir', type=str, default=DEFAULT_MIMIC_JPG_DIR,
                        help='Path to MIMIC-CXR JPEG directory')
    parser.add_argument('--h5_path', type=str, default=None,
                        help='Optional path to pre-converted H5 file (overrides JPEG)')
    parser.add_argument('--embedding_models', nargs='+',
                        default=['openai_3large'],
                        choices=list(EMBEDDING_MODELS.keys()),
                        help='Embedding models to evaluate')
    parser.add_argument('--max_concepts', type=int, default=0,
                        help='Max concepts to use (0 = all ~492k)')
    parser.add_argument('--image_batch_size', type=int, default=16,
                        help='Batch size for image encoding')
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
