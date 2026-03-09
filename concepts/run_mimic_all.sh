#!/bin/bash
# ============================================================
# Full MIMIC-CXR experiment pipeline (official splits)
#   Phase 1: Cache imaging features (GPU-heavy)
#   Phase 2: Linear probing for all models x both label sets
#   Phase 3: Concept importance analysis
#
# Usage:
#   nohup bash concepts/run_mimic_all.sh > concepts/results/mimic_run.log 2>&1 &
# ============================================================
set -e
cd /cbica/home/hanti/codes/clear_card
eval "$(micromamba shell hook --shell bash)" && micromamba activate ml311

MIMIC_JPG="/cbica/projects/CXR/data/MIMIC-CXR/data/images/mimic-cxr-jpg-2.1.0/mimic-cxr-jpg-2.1.0.physionet.org"
ARK_CKPT="checkpoints/Ark6_swinLarge768_ep50.pth.tar"
CXZ_CKPT="checkpoints/CheXzero_Models/best_64_0.0001_original_17000_0.863.pt"

echo "=========================================="
echo "MIMIC-CXR Full Pipeline: $(date)"
echo "=========================================="

# Helper to add model-specific checkpoint flags
fm_extra_args() {
    local fm="$1"
    if [ "$fm" = "ark_plus" ]; then
        echo "--ark_checkpoint $ARK_CKPT"
    elif [ "$fm" = "chexzero" ]; then
        echo "--chexzero_checkpoint $CXZ_CKPT"
    fi
}

# ==========================================================
# Phase 1: Cache all imaging features (GPU-heavy, sequential)
# ==========================================================
echo ""
echo "=========================================="
echo "PHASE 1: Feature caching — $(date)"
echo "=========================================="

# 1a. CLEAR model (caches CLIP image features + concept features)
echo ""
echo "[1/5] CLEAR (cache_only): $(date)"
python concepts/exp_linear_mimic.py \
    --dataset mimic \
    --embedding_models kalm_gemma3_12b \
    --mimic_jpg_dir "$MIMIC_JPG" \
    --cache_only
echo "CLEAR cache done: $(date)"

# 1b-1e. Foundation models (each caches its own features)
for FM in ark_plus rad_dino chexzero biomedclip; do
    echo ""
    echo "[FM: $FM] (cache_only): $(date)"
    python concepts/exp_linear_mimic_foundation.py \
        --model $FM \
        --mimic_jpg_dir "$MIMIC_JPG" \
        --cache_only \
        $(fm_extra_args $FM)
    echo "$FM cache done: $(date)"
done

echo ""
echo "Phase 1 complete: $(date)"
echo "=========================================="

# ==========================================================
# Phase 2: Linear probing (uses cached features, lighter GPU)
# ==========================================================
echo ""
echo "=========================================="
echo "PHASE 2: Linear probing — $(date)"
echo "=========================================="

# 2a. CLEAR — PheWAS (542 phenotypes)
echo ""
echo "[CLEAR / mimic PheWAS]: $(date)"
python concepts/exp_linear_mimic.py \
    --dataset mimic \
    --embedding_models kalm_gemma3_12b \
    --mimic_jpg_dir "$MIMIC_JPG" \
    --n_seeds 5
echo "CLEAR mimic PheWAS done: $(date)"

# 2b. CLEAR — 63-label
echo ""
echo "[CLEAR / mimic63]: $(date)"
python concepts/exp_linear_mimic.py \
    --dataset mimic63 \
    --embedding_models kalm_gemma3_12b \
    --mimic_jpg_dir "$MIMIC_JPG" \
    --n_seeds 5
echo "CLEAR mimic63 done: $(date)"

# 2c. Foundation models — PheWAS (542)
for FM in ark_plus rad_dino chexzero biomedclip; do
    echo ""
    echo "[FM: $FM / mimic PheWAS]: $(date)"
    python concepts/exp_linear_mimic_foundation.py \
        --dataset mimic \
        --model $FM \
        --mimic_jpg_dir "$MIMIC_JPG" \
        --n_seeds 5 \
        $(fm_extra_args $FM)
    echo "$FM mimic PheWAS done: $(date)"
done

# 2d. Foundation models — 63-label
for FM in ark_plus rad_dino chexzero biomedclip; do
    echo ""
    echo "[FM: $FM / mimic63]: $(date)"
    python concepts/exp_linear_mimic_foundation.py \
        --dataset mimic63 \
        --model $FM \
        --mimic_jpg_dir "$MIMIC_JPG" \
        --n_seeds 5 \
        $(fm_extra_args $FM)
    echo "$FM mimic63 done: $(date)"
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
    --results_dir concepts/results/linear_mimic_kalm_gemma3_12b \
    --top_k 100
echo "Concept importance (PheWAS) done: $(date)"

python concepts/concept_importance_mimic.py \
    --results_dir concepts/results/linear_mimic63_kalm_gemma3_12b \
    --top_k 100
echo "Concept importance (63-label) done: $(date)"

# AUROC comparison plot
python concepts/plot_auroc_comparison.py
echo "AUROC plot done: $(date)"

echo ""
echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
