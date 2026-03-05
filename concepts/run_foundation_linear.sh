#!/bin/bash
# ============================================================
# Run foundation model linear probing on MIMIC-CXR
# Execute from the repository root:
#   bash concepts/run_foundation_linear.sh
# ============================================================
set -e

# ---- Configuration (edit these) ----
LABELS_CSV="data/mimic_opportunistic_labels.csv"
MIMIC_JPG_DIR="/cbica/projects/CXR/data_p/mimic-cxr-jpg/"
N_SEEDS=5

# Optional: use H5 files instead of JPEG directory
# H5_TRAIN="/cbica/projects/CXR/data_p/mimic-cxr-jpg/mimic_train.h5"
# H5_VAL="/cbica/projects/CXR/data_p/mimic-cxr-jpg/mimic_validate.h5"
# H5_TEST="/cbica/projects/CXR/data_p/mimic-cxr-jpg/mimic_test.h5"
H5_ARGS=""
# H5_ARGS="--h5_train $H5_TRAIN --h5_val $H5_VAL --h5_test $H5_TEST"

# Model checkpoints (update paths after downloading)
ARK_CHECKPOINT="/cbica/projects/CXR/codes/clear_card/checkpoints/Ark6_swinLarge768_ep50.pth.tar"
CHEXZERO_CHECKPOINT="/cbica/projects/CXR/codes/clear_card/checkpoints/best_64_5e-05_original_22000_0.864.pt"

SCRIPT="concepts/exp_linear_mimic_foundation.py"

# ---- 1. RAD-DINO  (no checkpoint needed) ----
echo "============================================="
echo "[1/4]  RAD-DINO"
echo "============================================="
python $SCRIPT \
    --model rad_dino \
    --labels_csv $LABELS_CSV \
    --mimic_jpg_dir $MIMIC_JPG_DIR \
    --n_seeds $N_SEEDS \
    $H5_ARGS

# ---- 2. BiomedCLIP  (no checkpoint needed) ----
echo "============================================="
echo "[2/4]  BiomedCLIP"
echo "============================================="
python $SCRIPT \
    --model biomedclip \
    --labels_csv $LABELS_CSV \
    --mimic_jpg_dir $MIMIC_JPG_DIR \
    --n_seeds $N_SEEDS \
    $H5_ARGS

# ---- 3. CheXzero  (checkpoint required) ----
echo "============================================="
echo "[3/4]  CheXzero"
echo "============================================="
if [ -f "$CHEXZERO_CHECKPOINT" ]; then
    python $SCRIPT \
        --model chexzero \
        --chexzero_checkpoint $CHEXZERO_CHECKPOINT \
        --labels_csv $LABELS_CSV \
        --mimic_jpg_dir $MIMIC_JPG_DIR \
        --n_seeds $N_SEEDS \
        $H5_ARGS
else
    echo "  SKIP: CheXzero checkpoint not found at $CHEXZERO_CHECKPOINT"
fi

# ---- 4. Ark+  (checkpoint required) ----
echo "============================================="
echo "[4/4]  Ark+"
echo "============================================="
if [ -f "$ARK_CHECKPOINT" ]; then
    python $SCRIPT \
        --model ark_plus \
        --ark_checkpoint $ARK_CHECKPOINT \
        --labels_csv $LABELS_CSV \
        --mimic_jpg_dir $MIMIC_JPG_DIR \
        --n_seeds $N_SEEDS \
        $H5_ARGS
else
    echo "  SKIP: Ark+ checkpoint not found at $ARK_CHECKPOINT"
fi

echo ""
echo "============================================="
echo "All foundation model experiments complete."
echo "Results in: concepts/results/linear_mimic_foundation_*/"
echo "============================================="
