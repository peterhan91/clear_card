#!/bin/bash
# ============================================================
# Full CheXchoNet experiment pipeline
#   Phase 1: Cache imaging features (GPU-heavy)
#   Phase 2: Linear probing for all 5 models
#   Phase 3: Concept importance analysis
#
# 6 binary labels, 50K/7K/14K splits, H5 at 224x224
#
# Usage:
#   nohup bash concepts/run_chexchonet_all.sh > concepts/results/chexchonet_run.log 2>&1 &
# ============================================================
set -e
cd /cbica/home/hanti/codes/clear_card
eval "$(micromamba shell hook --shell bash)" && micromamba activate ml311

DATASET="chexchonet"
N_SEEDS=5
ARK_CKPT="checkpoints/Ark6_swinLarge768_ep50.pth.tar"

echo "=========================================="
echo "CheXchoNet Full Pipeline: $(date)"
echo "=========================================="

# ==========================================================
# Phase 1: Cache all imaging features (GPU-heavy, sequential)
# ==========================================================
echo ""
echo "=========================================="
echo "PHASE 1: Feature caching — $(date)"
echo "=========================================="

# 1a. CLEAR model
echo ""
echo "[1/5] CLEAR (cache_only): $(date)"
python concepts/exp_linear_mimic.py \
    --dataset $DATASET \
    --embedding_models kalm_gemma3_12b \
    --cache_only
echo "CLEAR cache done: $(date)"

# 1b-1e. Foundation models
for FM in rad_dino biomedclip chexzero ark_plus; do
    echo ""
    echo "[FM: $FM] (cache_only): $(date)"
    CMD="python concepts/exp_linear_mimic_foundation.py \
        --model $FM \
        --dataset $DATASET \
        --cache_only"
    if [ "$FM" = "ark_plus" ]; then
        CMD="$CMD --ark_checkpoint $ARK_CKPT"
    fi
    eval $CMD
    echo "$FM cache done: $(date)"
done

echo ""
echo "Phase 1 complete: $(date)"
echo "=========================================="

# ==========================================================
# Phase 2: Linear probing (uses cached features)
# ==========================================================
echo ""
echo "=========================================="
echo "PHASE 2: Linear probing — $(date)"
echo "=========================================="

# 2a. CLEAR
echo ""
echo "[CLEAR / chexchonet]: $(date)"
python concepts/exp_linear_mimic.py \
    --dataset $DATASET \
    --embedding_models kalm_gemma3_12b \
    --n_seeds $N_SEEDS
echo "CLEAR done: $(date)"

# 2b. Foundation models
for FM in rad_dino biomedclip chexzero ark_plus; do
    echo ""
    echo "[FM: $FM / chexchonet]: $(date)"
    CMD="python concepts/exp_linear_mimic_foundation.py \
        --model $FM \
        --dataset $DATASET \
        --n_seeds $N_SEEDS"
    if [ "$FM" = "ark_plus" ]; then
        CMD="$CMD --ark_checkpoint $ARK_CKPT"
    fi
    eval $CMD
    echo "$FM done: $(date)"
done

echo ""
echo "Phase 2 complete: $(date)"
echo "=========================================="

# ==========================================================
# Phase 3: Concept importance analysis
# ==========================================================
echo ""
echo "=========================================="
echo "PHASE 3: Concept importance — $(date)"
echo "=========================================="

python concepts/concept_importance_mimic.py \
    --results_dir concepts/results/linear_chexchonet_kalm_gemma3_12b \
    --top_k 100
echo "Concept importance done: $(date)"

# Verify results
echo ""
echo "=========================================="
echo "Results verification:"
echo "=========================================="
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
echo "ALL DONE: $(date)"
echo "=========================================="
