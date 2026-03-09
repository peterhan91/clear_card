#!/bin/bash
# ============================================================
# Run remaining MIMIC-63 experiments:
#   - CheXzero feature caching (missing entirely)
#   - All 4 FMs linear probing on mimic63
#   - CLEAR mimic63 already done — skip
#
# Usage:
#   nohup bash concepts/run_mimic63_remaining.sh > concepts/results/mimic63_run.log 2>&1 &
# ============================================================
set -e
cd /cbica/home/hanti/codes/clear_card
eval "$(micromamba shell hook --shell bash)" && micromamba activate ml311

MIMIC_JPG="/cbica/projects/CXR/data/MIMIC-CXR/data/images/mimic-cxr-jpg-2.1.0/mimic-cxr-jpg-2.1.0.physionet.org"
ARK_CKPT="checkpoints/Ark6_swinLarge768_ep50.pth.tar"
CXZ_CKPT="checkpoints/CheXzero_Models/best_64_0.0001_original_17000_0.863.pt"

echo "=========================================="
echo "MIMIC-63 Remaining Experiments: $(date)"
echo "=========================================="

# Step 1: Cache CheXzero features (missing for MIMIC)
echo ""
echo "[Step 1] CheXzero feature caching: $(date)"
python concepts/exp_linear_mimic_foundation.py \
    --dataset mimic \
    --model chexzero \
    --chexzero_checkpoint "$CXZ_CKPT" \
    --mimic_jpg_dir "$MIMIC_JPG" \
    --cache_only
echo "CheXzero cache done: $(date)"

# Step 2: FM linear probing on mimic63 (reuses cached mimic features)
for FM in ark_plus rad_dino chexzero biomedclip; do
    echo ""
    echo "[FM: $FM / mimic63]: $(date)"
    EXTRA=""
    if [ "$FM" = "ark_plus" ]; then
        EXTRA="--ark_checkpoint $ARK_CKPT"
    elif [ "$FM" = "chexzero" ]; then
        EXTRA="--chexzero_checkpoint $CXZ_CKPT"
    fi
    python concepts/exp_linear_mimic_foundation.py \
        --dataset mimic63 \
        --model $FM \
        --mimic_jpg_dir "$MIMIC_JPG" \
        --n_seeds 5 \
        $EXTRA
    echo "$FM mimic63 done: $(date)"
done

echo ""
echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
