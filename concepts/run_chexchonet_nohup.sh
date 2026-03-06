#!/bin/bash
# ============================================================
# Run CLEAR + 4 FM linear probing on CheXchoNet
# 6 binary classification labels, 50K/7K/14K splits, H5 at 224x224
#
# Usage:
#   nohup bash concepts/run_chexchonet_nohup.sh > concepts/results/chexchonet_run.log 2>&1 &
# ============================================================
set -e

cd /cbica/home/hanti/codes/clear_card
eval "$(micromamba shell hook --shell bash)" && micromamba activate ml311

# ---- Paths ----
DATASET="chexchonet"
N_SEEDS=5
ARK_CKPT="checkpoints/Ark6_swinLarge768_ep50.pth.tar"
CXZ_CKPT="checkpoints/CheXzero_Models/best_64_0.0001_original_17000_0.863.pt"

echo "=========================================="
echo "Start: $(date)"
echo "Dataset: $DATASET (6 binary classification labels)"
echo "=========================================="

# ---- 1. CLEAR (concept-based, ViT-7B LoRA) ----
echo ""
echo "[1/5] CLEAR  (ViT-7B LoRA, feat_dim=3840 via KaLM-Gemma3-12B)"
python concepts/exp_linear_mimic.py \
    --dataset $DATASET \
    --embedding_models kalm_gemma3_12b \
    --n_seeds $N_SEEDS
echo "[1/5] CLEAR done: $(date)"

# ---- 2. RAD-DINO ----
FM_SCRIPT="concepts/exp_linear_mimic_foundation.py"
FM_COMMON="--dataset $DATASET --n_seeds $N_SEEDS"

echo ""
echo "[2/5] RAD-DINO  (feat_dim=768)"
python $FM_SCRIPT --model rad_dino $FM_COMMON
echo "[2/5] RAD-DINO done: $(date)"

# ---- 3. BiomedCLIP ----
echo ""
echo "[3/5] BiomedCLIP  (feat_dim=512)"
python $FM_SCRIPT --model biomedclip $FM_COMMON
echo "[3/5] BiomedCLIP done: $(date)"

# ---- 4. CheXzero ----
echo ""
echo "[4/5] CheXzero  (feat_dim=512)"
python $FM_SCRIPT --model chexzero --chexzero_checkpoint "$CXZ_CKPT" $FM_COMMON
echo "[4/5] CheXzero done: $(date)"

# ---- 5. Ark+ ----
echo ""
echo "[5/5] Ark+  (feat_dim=1376)"
python $FM_SCRIPT --model ark_plus --ark_checkpoint "$ARK_CKPT" $FM_COMMON
echo "[5/5] Ark+ done: $(date)"

echo ""
echo "=========================================="
echo "All 5 models complete: $(date)"
echo "=========================================="

# ---- Verify results ----
echo ""
echo "Results verification:"
for dir in linear_chexchonet_kalm_gemma3_12b \
    linear_chexchonet_foundation_rad_dino \
    linear_chexchonet_foundation_biomedclip \
    linear_chexchonet_foundation_chexzero \
    linear_chexchonet_foundation_ark_plus; do
    count=$(ls concepts/results/$dir/predictions/seed_*_predictions.pkl 2>/dev/null | wc -l)
    echo "  $dir: $count/5 seeds"
done

echo ""
echo "=========================================="
echo "All complete: $(date)"
echo "=========================================="
