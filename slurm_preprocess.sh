#!/bin/bash
#SBATCH --job-name=cxr_preprocess
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=preprocess_%j.out
#SBATCH --error=preprocess_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu

source /cbica/projects/CXR/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

REPO_PATH="/cbica/projects/CXR/codes/clear_card"
cd "$REPO_PATH"

# ============================================
# Path Configuration
# ============================================
# DATASET: chexpert-plus, rexgradient, or both
DATASET="${DATASET:-chexpert-plus}"

# DATA_DIR: where CSV files are located
DATA_DIR="${DATA_DIR:-data}"

# OUTPUT_BASE: base directory for H5 output (subdirs created per dataset)
OUTPUT_BASE="${OUTPUT_BASE:-/cbica/projects/CXR/data_p}"

# IMAGE_ROOT: prepended to image paths in CSV
# - CheXpert: set to parent of train/valid/test folders (PNG/PNG contains train/valid/test)
# - ReXGradient: paths are remapped in code, so this is ignored
CHEXPERT_IMAGE_ROOT="${CHEXPERT_IMAGE_ROOT:-/cbica/projects/CXR/data/CheXpert/chexpertplus/PNG/PNG}"
REXGRADIENT_IMAGE_ROOT="${REXGRADIENT_IMAGE_ROOT:-}"

# ============================================
# Processing Configuration
# ============================================
RESOLUTION="${RESOLUTION:-448}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-16}"

echo "========================================"
echo "CXR Preprocessing Job"
echo "========================================"
echo "Dataset:              $DATASET"
echo "Data dir (CSVs):      $DATA_DIR"
echo "Output base:          $OUTPUT_BASE"
echo "CheXpert image root:  $CHEXPERT_IMAGE_ROOT"
echo "ReXGradient img root: $REXGRADIENT_IMAGE_ROOT"
echo "Resolution:           $RESOLUTION"
echo "Seed:                 $SEED"
echo "Num workers:          $NUM_WORKERS"
echo "========================================"

# Create output directories
CHEXPERT_OUTPUT_DIR="${OUTPUT_BASE}/chexpert-plus"
REXGRADIENT_OUTPUT_DIR="${OUTPUT_BASE}/rexgradient"
mkdir -p "$CHEXPERT_OUTPUT_DIR"
mkdir -p "$REXGRADIENT_OUTPUT_DIR"

# ============================================
# Preprocessing
# ============================================
if [[ "$DATASET" == "chexpert-plus" || "$DATASET" == "both" ]]; then
    echo ""
    echo ">>> Processing CheXpert-Plus..."
    echo ">>> Output: $CHEXPERT_OUTPUT_DIR"
    python preprocessing/preprocess_splits.py \
        --dataset chexpert-plus \
        --image_root "$CHEXPERT_IMAGE_ROOT" \
        --data_dir "$DATA_DIR" \
        --output_dir "$CHEXPERT_OUTPUT_DIR" \
        --resolution "$RESOLUTION" \
        --num_workers "$NUM_WORKERS"
fi

if [[ "$DATASET" == "rexgradient" || "$DATASET" == "both" ]]; then
    echo ""
    echo ">>> Processing ReXGradient..."
    echo ">>> Output: $REXGRADIENT_OUTPUT_DIR"
    python preprocessing/preprocess_splits.py \
        --dataset rexgradient \
        --image_root "$REXGRADIENT_IMAGE_ROOT" \
        --data_dir "$DATA_DIR" \
        --output_dir "$REXGRADIENT_OUTPUT_DIR" \
        --resolution "$RESOLUTION" \
        --seed "$SEED" \
        --num_workers "$NUM_WORKERS"
fi

# ============================================
# Verification
# ============================================
echo ""
echo "========================================"
echo "Verifying patient splits..."
echo "========================================"
if [[ "$DATASET" == "chexpert-plus" || "$DATASET" == "both" ]]; then
    python preprocessing/preprocess_splits.py \
        --dataset chexpert-plus \
        --output_dir "$CHEXPERT_OUTPUT_DIR" \
        --verify_only
fi
if [[ "$DATASET" == "rexgradient" || "$DATASET" == "both" ]]; then
    python preprocessing/preprocess_splits.py \
        --dataset rexgradient \
        --output_dir "$REXGRADIENT_OUTPUT_DIR" \
        --verify_only
fi

echo ""
echo "========================================"
echo "Verifying H5-CSV pairing..."
echo "========================================"
if [[ "$DATASET" == "chexpert-plus" || "$DATASET" == "both" ]]; then
    python preprocessing/preprocess_splits.py \
        --dataset chexpert-plus \
        --output_dir "$CHEXPERT_OUTPUT_DIR" \
        --verify_pairing
fi
if [[ "$DATASET" == "rexgradient" || "$DATASET" == "both" ]]; then
    python preprocessing/preprocess_splits.py \
        --dataset rexgradient \
        --output_dir "$REXGRADIENT_OUTPUT_DIR" \
        --verify_pairing
fi

echo ""
echo "========================================"
echo "Done! Output saved to: $OUTPUT_BASE"
echo "========================================"
if [[ "$DATASET" == "chexpert-plus" || "$DATASET" == "both" ]]; then
    echo "CheXpert-Plus:"
    ls -lh "$CHEXPERT_OUTPUT_DIR"
fi
if [[ "$DATASET" == "rexgradient" || "$DATASET" == "both" ]]; then
    echo "ReXGradient:"
    ls -lh "$REXGRADIENT_OUTPUT_DIR"
fi
