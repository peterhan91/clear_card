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
conda activate axolotl

REPO_PATH="/cbica/projects/CXR/codes/clear_card"
cd "$REPO_PATH"

# ============================================
# Path Configuration
# ============================================
# DATASET: chexpert-plus, rexgradient, or both
DATASET="${DATASET:-chexpert-plus}"

# DATA_DIR: where CSV files are located
DATA_DIR="${DATA_DIR:-data}"

# OUTPUT_ROOT: root directory for preprocessed output (each dataset gets a subdirectory)
OUTPUT_ROOT="${OUTPUT_ROOT:-/cbica/projects/CXR/data_p}"
CHEXPERT_OUTPUT_DIR="${CHEXPERT_OUTPUT_DIR:-${OUTPUT_ROOT}/chexpert-plus}"
REXGRADIENT_OUTPUT_DIR="${REXGRADIENT_OUTPUT_DIR:-${OUTPUT_ROOT}/rexgradient}"

# IMAGE_ROOT: prepended to image paths in CSV
# - CheXpert: set to parent of train/valid/test folders
# - ReXGradient: set to "" if CSV has absolute paths
CHEXPERT_IMAGE_ROOT="${CHEXPERT_IMAGE_ROOT:-/cbica/projects/CXR/chexpert}"
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
echo "Output root:          $OUTPUT_ROOT"
echo "CheXpert output:      $CHEXPERT_OUTPUT_DIR"
echo "ReXGradient output:   $REXGRADIENT_OUTPUT_DIR"
echo "CheXpert image root:  $CHEXPERT_IMAGE_ROOT"
echo "ReXGradient img root: $REXGRADIENT_IMAGE_ROOT"
echo "Resolution:           $RESOLUTION"
echo "Seed:                 $SEED"
echo "Num workers:          $NUM_WORKERS"
echo "========================================"

# Create output directories
mkdir -p "$CHEXPERT_OUTPUT_DIR" "$REXGRADIENT_OUTPUT_DIR"

# ============================================
# Preprocessing
# ============================================
if [[ "$DATASET" == "chexpert-plus" || "$DATASET" == "both" ]]; then
    echo ""
    echo ">>> Processing CheXpert-Plus..."
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
if [[ "$DATASET" == "chexpert-plus" || "$DATASET" == "both" ]]; then
    echo ""
    echo "========================================"
    echo "Verifying CheXpert-Plus splits..."
    echo "========================================"
    python preprocessing/preprocess_splits.py \
        --dataset chexpert-plus \
        --output_dir "$CHEXPERT_OUTPUT_DIR" \
        --verify_only
    python preprocessing/preprocess_splits.py \
        --dataset chexpert-plus \
        --output_dir "$CHEXPERT_OUTPUT_DIR" \
        --verify_pairing
    echo "Output:"
    ls -lh "$CHEXPERT_OUTPUT_DIR"
fi

if [[ "$DATASET" == "rexgradient" || "$DATASET" == "both" ]]; then
    echo ""
    echo "========================================"
    echo "Verifying ReXGradient splits..."
    echo "========================================"
    python preprocessing/preprocess_splits.py \
        --dataset rexgradient \
        --output_dir "$REXGRADIENT_OUTPUT_DIR" \
        --verify_only
    python preprocessing/preprocess_splits.py \
        --dataset rexgradient \
        --output_dir "$REXGRADIENT_OUTPUT_DIR" \
        --verify_pairing
    echo "Output:"
    ls -lh "$REXGRADIENT_OUTPUT_DIR"
fi

echo ""
echo "========================================"
echo "Done! Output saved to: $OUTPUT_ROOT"
echo "========================================"
