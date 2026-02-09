# CLEAR-CARD: CXR CLIP Training Pipeline

Preprocessing and contrastive learning pipeline for CheXpert-Plus and ReXGradient chest X-ray datasets with patient-level train/val/test splits.

## Structure

```
clear_card/
├── data/                          # CSV metadata files
│   ├── chexpert_train.csv         # CheXpert-Plus training (64,525 patients)
│   ├── chexpert_valid.csv         # Official CheXpert validation (200 patients)
│   ├── chexpert_test.csv          # Official CheXpert test (500 patients)
│   └── rexgradient_all.csv        # ReXGradient full dataset (109,487 patients)
├── preprocessing/
│   ├── data_process.py            # Core H5 conversion (single & parallel)
│   ├── preprocess_splits.py       # Patient-level split preprocessing
│   ├── run_preprocess_legacy.py   # Legacy multi-dataset preprocessing (reference only)
│   └── data_utils/dcm_png.py      # DICOM to PNG conversion
├── training/
│   ├── model.py                   # CLIP model architecture (ViT-B + text transformer)
│   ├── clip.py                    # CLIP loading utilities & tokenization
│   ├── simple_tokenizer.py        # BPE tokenizer
│   ├── bpe_simple_vocab_16e6.txt.gz  # BPE vocabulary
│   ├── train.py                   # Dataset classes, data loading, text preprocessing
│   ├── eval.py                    # Evaluation (AUC, bootstrap, confidence intervals)
│   ├── zero_shot.py               # Zero-shot evaluation pipeline
│   ├── run_train.py               # Basic single-GPU training
│   └── run_train_improved.py      # Advanced training (DDP, validation, early stopping)
├── slurm_preprocess.sh            # Slurm preprocessing job
└── slurm_train.sh                 # Slurm training job
```

## Cluster Data Layout (CUBIC)

```
/cbica/projects/CXR/data_p/
├── chexpert-plus/
│   ├── chexpert_plus_train.h5
│   ├── chexpert_plus_train_metadata.csv
│   ├── chexpert_plus_valid.h5
│   ├── chexpert_plus_valid_metadata.csv
│   ├── chexpert_plus_test.h5
│   └── chexpert_plus_test_metadata.csv
└── rexgradient/
    ├── rexgradient_train.h5
    ├── rexgradient_train_metadata.csv
    ├── rexgradient_valid.h5
    ├── rexgradient_valid_metadata.csv
    ├── rexgradient_test.h5
    └── rexgradient_test_metadata.csv
```

## Preprocessing

### Command Line

```bash
python preprocessing/preprocess_splits.py \
    --dataset <chexpert-plus|rexgradient|both> \
    --image_root /path/to/images \
    --data_dir data \
    --output_dir /path/to/output \
    --resolution 448 \
    --num_workers 8
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | `chexpert-plus`, `rexgradient`, or `both` |
| `--image_root` | `""` | Root path prepended to image paths in CSV |
| `--data_dir` | `data` | Directory containing input CSV files |
| `--output_dir` | `data/h5` | Directory for output H5 files |
| `--resolution` | `448` | Target image resolution |
| `--num_workers` | auto | Number of parallel workers |
| `--seed` | `42` | Random seed for ReXGradient split |

### Slurm Preprocessing

```bash
# CheXpert-Plus only
sbatch slurm_preprocess.sh

# ReXGradient only
DATASET=rexgradient sbatch slurm_preprocess.sh

# Both datasets
DATASET=both sbatch slurm_preprocess.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `chexpert-plus` | `chexpert-plus`, `rexgradient`, or `both` |
| `OUTPUT_ROOT` | `/cbica/projects/CXR/data_p` | Root for preprocessed output |
| `CHEXPERT_IMAGE_ROOT` | `/cbica/projects/CXR/chexpert` | Root for CheXpert images |
| `REXGRADIENT_IMAGE_ROOT` | `""` | Root for ReXGradient images |

### Verify No Patient Overlap

```bash
python preprocessing/preprocess_splits.py --dataset chexpert-plus \
    --output_dir /cbica/projects/CXR/data_p/chexpert-plus --verify_only
python preprocessing/preprocess_splits.py --dataset rexgradient \
    --output_dir /cbica/projects/CXR/data_p/rexgradient --verify_only
```

## Training

### CXR CLIP Model

Train a contrastive vision-language model (CheXzero-style) on preprocessed CXR images paired with radiology report impressions.

### Quick Start (CUBIC)

```bash
# Default: CheXpert-Plus single dataset, with validation + early stopping
sbatch slurm_train.sh

# With DINOv3 vision encoder
USE_DINOV3=1 sbatch slurm_train.sh

# With DINOv3 ViT-L/16
USE_DINOV3=1 DINOV3_MODEL=dinov3_vitl16 sbatch slurm_train.sh

# Multi-dataset (CheXpert-Plus + ReXGradient)
USE_MULTI=1 sbatch slurm_train.sh
```

### Command Line

```bash
# Basic training (single GPU)
python training/run_train.py \
    --cxr_filepath /cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_train.h5 \
    --txt_filepath /cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_train_metadata.csv \
    --batch_size 64 --epochs 40 --lr 1e-4

# Advanced training with validation and early stopping
python training/run_train_improved.py \
    --cxr_filepath /cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_train.h5 \
    --txt_filepath /cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_train_metadata.csv \
    --val_cxr_filepath /cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_valid.h5 \
    --val_label_path data/chexpert_valid.csv \
    --do_validate --early_stopping --test_after_training \
    --batch_size 64 --epochs 40 --lr 1e-4

# Multi-GPU DDP training (2xA100)
torchrun --nproc_per_node=2 training/run_train_improved.py \
    --use_ddp --use_multi_datasets \
    --dataset_paths \
        /cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_train.h5,/cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_train_metadata.csv \
        /cbica/projects/CXR/data_p/rexgradient/rexgradient_train.h5,/cbica/projects/CXR/data_p/rexgradient/rexgradient_train_metadata.csv \
    --do_validate --early_stopping

# DINOv3 as vision encoder
python training/run_train_improved.py \
    --use_dinov3 --dinov3_model_name dinov3_vitb16 \
    --dinov3_repo_dir /cbica/projects/CXR/codes/dinov3 \
    --dinov3_weights /cbica/projects/CXR/codes/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    --cxr_filepath /cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_train.h5 \
    --txt_filepath /cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_train_metadata.csv \
    --do_validate --early_stopping
```

### Training Arguments (run_train_improved.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--cxr_filepath` | `.../chexpert-plus/chexpert_plus_train.h5` | Training H5 images |
| `--txt_filepath` | `.../chexpert-plus/chexpert_plus_train_metadata.csv` | Training metadata CSV |
| `--use_multi_datasets` | off | Train on multiple datasets |
| `--dataset_paths` | CheXpert+ReXGradient | `img.h5,text.csv` pairs |
| `--batch_size` | `64` | Batch size per GPU |
| `--epochs` | `40` | Max training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--weight_decay` | `0.2` | AdamW weight decay |
| `--lr_schedule` | `cosine` | LR schedule (cosine with warmup) |
| `--grad_accum_steps` | `4` | Gradient accumulation steps |
| `--do_validate` | off | Validate during training |
| `--early_stopping` | off | Enable early stopping |
| `--patience` | `5` | Early stopping patience |
| `--use_dinov3` | off | Use DINOv3 vision encoder |
| `--dinov3_model_name` | `dinov3_vitb16` | DINOv3 variant |
| `--dinov3_repo_dir` | `/cbica/.../dinov3` | Path to local DINOv3 repo |
| `--dinov3_weights` | `.../dinov3_vitb16_...pth` | Path to DINOv3 weights |
| `--freeze_dinov3` | off | Freeze DINOv3 backbone |
| `--use_ddp` | off | Multi-GPU DDP training |
| `--test_after_training` | off | Run zero-shot test after training |

### Note on Multi-Dataset Column Names

CheXpert metadata uses `impression` (lowercase) while ReXGradient uses `Impression` (uppercase). For multi-dataset training, normalize column names first:

```bash
python -c "
import pandas as pd
df = pd.read_csv('/cbica/projects/CXR/data_p/rexgradient/rexgradient_train_metadata.csv')
df.rename(columns={'Impression': 'impression'}, inplace=True)
df.to_csv('/cbica/projects/CXR/data_p/rexgradient/rexgradient_train_metadata.csv', index=False)
"
```

### Model Architecture

- **Vision**: ViT-B/16 (12 layers, 768 width, 16px patches) or DINOv3
- **Text**: Transformer (12 layers, 512 width, 8 heads, context length 77)
- **Embedding**: 768-dim shared space
- **Loss**: Symmetric contrastive (CrossEntropy on image-text similarity matrix)

## Preprocessing Pipeline

```
Original Image (e.g., 2000x1500 RGB)
         │
         ▼
┌─────────────────┐
│ Load (cv2)      │  Parallel workers
│ BGR → RGB       │
│ Resize (LANCZOS)│  Preserve aspect ratio
│ Grayscale       │
│ Zero-pad        │  Center in square
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ H5 File         │  Sequential write
│ [N, H, W]       │  float32
│ Dataset: 'cxr'  │
└─────────────────┘
```

## Requirements

```
# Preprocessing
numpy
pandas
h5py
opencv-python
pillow
tqdm

# Training
torch
torchvision
scikit-learn
scipy
ftfy
regex
matplotlib
```
