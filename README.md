# CXR Preprocessing Pipeline

Preprocessing pipeline for CheXpert-Plus and ReXGradient chest X-ray datasets with patient-level train/val/test splits.

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
│   └── data_utils/dcm_png.py      # DICOM to PNG conversion
└── slurm_preprocess.sh            # Slurm job script
```

## Usage

### Command Line

```bash
python preprocessing/preprocess_splits.py \
    --dataset <chexpert-plus|rexgradient|both> \
    --image_root /path/to/images \
    --data_dir data \
    --output_dir data/h5 \
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

### How Paths Work

The `--image_root` is prepended to paths in CSV:

```
# CheXpert-Plus
CSV path:    train/patient42142/study5/view1_frontal.jpg
image_root:  /data/chexpert
Final path:  /data/chexpert/train/patient42142/study5/view1_frontal.jpg

# ReXGradient (if CSV has absolute paths, use empty image_root)
CSV path:    /home/user/.cache/huggingface/.../image.png
image_root:  ""
Final path:  /home/user/.cache/huggingface/.../image.png
```

## Slurm

### Basic Usage

```bash
# CheXpert-Plus only
sbatch slurm_preprocess.sh

# ReXGradient only
DATASET=rexgradient sbatch slurm_preprocess.sh

# Both datasets
DATASET=both sbatch slurm_preprocess.sh
```

### Custom Paths

```bash
DATASET=both \
CHEXPERT_IMAGE_ROOT=/cbica/projects/CXR/chexpert \
REXGRADIENT_IMAGE_ROOT="" \
OUTPUT_DIR=/cbica/projects/CXR/h5_output \
sbatch slurm_preprocess.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `chexpert-plus` | `chexpert-plus`, `rexgradient`, or `both` |
| `CHEXPERT_IMAGE_ROOT` | `/cbica/projects/CXR/chexpert` | Root for CheXpert images |
| `REXGRADIENT_IMAGE_ROOT` | `""` | Root for ReXGradient images |
| `OUTPUT_DIR` | `/cbica/projects/CXR/h5_output` | Where to save H5 files |
| `DATA_DIR` | `data` | Where CSV files are located |
| `NUM_WORKERS` | `16` | Parallel workers |
| `RESOLUTION` | `448` | Image resolution |
| `SEED` | `42` | Random seed for ReXGradient |

### Verify No Patient Overlap

```bash
python preprocessing/preprocess_splits.py --dataset both --output_dir data/h5 --verify_only
```

## Output

Creates H5 files and metadata CSVs with guaranteed no patient overlap between splits:

```
OUTPUT_DIR/
├── chexpert_plus_train.h5
├── chexpert_plus_train_metadata.csv
├── chexpert_plus_valid.h5
├── chexpert_plus_valid_metadata.csv
├── chexpert_plus_test.h5
├── chexpert_plus_test_metadata.csv
├── rexgradient_train.h5
├── rexgradient_train_metadata.csv
├── rexgradient_valid.h5
├── rexgradient_valid_metadata.csv
├── rexgradient_test.h5
└── rexgradient_test_metadata.csv
```

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
numpy
pandas
h5py
opencv-python
pillow
tqdm
```
