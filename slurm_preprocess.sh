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

# OUTPUT_DIR: where H5 and metadata files will be saved
OUTPUT_DIR="${OUTPUT_DIR:-/cbica/projects/CXR/h5_output}"

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
echo "Output dir (H5):      $OUTPUT_DIR"
echo "CheXpert image root:  $CHEXPERT_IMAGE_ROOT"
echo "ReXGradient img root: $REXGRADIENT_IMAGE_ROOT"
echo "Resolution:           $RESOLUTION"
echo "Seed:                 $SEED"
echo "Num workers:          $NUM_WORKERS"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

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
        --output_dir "$OUTPUT_DIR" \
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
        --output_dir "$OUTPUT_DIR" \
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
python preprocessing/preprocess_splits.py \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --verify_only

echo ""
echo "========================================"
echo "Verifying H5-CSV pairing..."
echo "========================================"
python preprocessing/preprocess_splits.py \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --verify_pairing

echo ""
echo "========================================"
echo "Done! Output saved to: $OUTPUT_DIR"
echo "========================================"
ls -lh "$OUTPUT_DIR"
