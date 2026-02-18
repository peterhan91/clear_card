#!/bin/bash
# =============================================================================
# launch_dinov3_sweep.sh
#
# Wrapper script that submits 3 SLURM training jobs via slurm_train.sh:
#   1. CheXpert-Plus + ReXGradient (combined)
#   2. CheXpert-Plus only
#   3. ReXGradient only
#
# All jobs use DINOv3 + LoRA with validation, early stopping, and final testing
# to select the best performing model.
#
# Usage:
#   ./launch_dinov3_sweep.sh --model dinov3_vith16plus --lora_rank 8
#   ./launch_dinov3_sweep.sh --model dinov3_vit7b16 --lora_rank 16 --lora_alpha 32
#   ./launch_dinov3_sweep.sh --model dinov3_vitl16 --no_lora
# =============================================================================

set -euo pipefail

# ============================================
# Defaults
# ============================================
DINOV3_MODEL="dinov3_vith16plus"
USE_LORA=1
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES=""
EPOCHS=40
LR=1e-4
GRAD_ACCUM=1
SEED=1234
DRY_RUN=0

# ============================================
# Parse arguments
# ============================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          DINOV3_MODEL="$2";        shift 2 ;;
        --no_lora)        USE_LORA=0;               shift   ;;
        --lora_rank)      LORA_RANK="$2";           shift 2 ;;
        --lora_alpha)     LORA_ALPHA="$2";          shift 2 ;;
        --lora_dropout)   LORA_DROPOUT="$2";        shift 2 ;;
        --lora_targets)   LORA_TARGET_MODULES="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2";              shift 2 ;;
        --lr)             LR="$2";                  shift 2 ;;
        --grad_accum)     GRAD_ACCUM="$2";          shift 2 ;;
        --seed)           SEED="$2";                shift 2 ;;
        --dry_run)        DRY_RUN=1;                shift   ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL       DINOv3 variant (default: dinov3_vith16plus)"
            echo "                      Choices: dinov3_vits16, dinov3_vits16plus, dinov3_vitb16,"
            echo "                               dinov3_vitl16, dinov3_vith16plus, dinov3_vit7b16"
            echo "  --no_lora           Disable LoRA (full finetune; not allowed for vit7b16)"
            echo "  --lora_rank N       LoRA rank (default: 16)"
            echo "  --lora_alpha N      LoRA alpha (default: 32)"
            echo "  --lora_dropout F    LoRA dropout (default: 0.05)"
            echo "  --lora_targets M    Space-separated target modules (default: auto-detect)"
            echo "  --epochs N          Training epochs (default: 40)"
            echo "  --lr F              Learning rate (default: 1e-4)"
            echo "  --grad_accum N      Gradient accumulation steps (default: 1)"
            echo "  --seed N            Random seed (default: 1234)"
            echo "  --dry_run           Print sbatch commands without submitting"
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Run $0 --help for usage"
            exit 1
            ;;
    esac
done

# ============================================
# Validate
# ============================================
VALID_MODELS="dinov3_vits16 dinov3_vits16plus dinov3_vitb16 dinov3_vitl16 dinov3_vith16plus dinov3_vit7b16"
if ! echo "$VALID_MODELS" | grep -qw "$DINOV3_MODEL"; then
    echo "Error: Invalid model '$DINOV3_MODEL'"
    echo "Valid choices: $VALID_MODELS"
    exit 1
fi

if [ "$DINOV3_MODEL" == "dinov3_vit7b16" ] && [ "$USE_LORA" == "0" ]; then
    echo "Error: dinov3_vit7b16 is too large for full finetuning. LoRA is required."
    echo "Remove --no_lora to enable LoRA."
    exit 1
fi

# Build tag for model name
VIT_TAG="${DINOV3_MODEL#dinov3_}"  # e.g., vith16plus, vit7b16
if [ "$USE_LORA" == "1" ]; then
    LORA_TAG="lora-r${LORA_RANK}-a${LORA_ALPHA}"
else
    LORA_TAG="full"
fi

# ============================================
# Print summary
# ============================================
echo "========================================================"
echo "  DINOv3 Dataset Sweep"
echo "========================================================"
echo "  Model:        $DINOV3_MODEL"
echo "  Mode:         $([ "$USE_LORA" == "1" ] && echo "LoRA (rank=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT)" || echo "Full finetune")"
echo "  Epochs:       $EPOCHS"
echo "  LR:           $LR"
echo "  Grad accum:   $GRAD_ACCUM"
echo "  Seed:         $SEED"
echo ""
echo "  Jobs to submit:"
echo "    1. CheXpert-Plus + ReXGradient (combined)"
echo "    2. CheXpert-Plus only"
echo "    3. ReXGradient only"
echo ""
echo "  All jobs: DDP=2xA100, auto batch size, validation,"
echo "            early stopping, test after training"
echo "========================================================"
echo ""

# ============================================
# Data paths
# ============================================
DATA_ROOT="/cbica/projects/CXR/data_p"
CHEXPERT_DIR="${DATA_ROOT}/chexpert-plus"
REXGRADIENT_DIR="${DATA_ROOT}/rexgradient"

# ============================================
# Common env vars for all 3 jobs
# ============================================
# Uses "VAR=val sbatch script.sh" pattern so sbatch inherits the environment.
COMMON_ENV="USE_DINOV3=1"
COMMON_ENV+=" DINOV3_MODEL=$DINOV3_MODEL"
COMMON_ENV+=" USE_LORA=$USE_LORA"
COMMON_ENV+=" LORA_RANK=$LORA_RANK"
COMMON_ENV+=" LORA_ALPHA=$LORA_ALPHA"
COMMON_ENV+=" LORA_DROPOUT=$LORA_DROPOUT"
if [ -n "$LORA_TARGET_MODULES" ]; then
    COMMON_ENV+=" LORA_TARGET_MODULES=$LORA_TARGET_MODULES"
fi
COMMON_ENV+=" EPOCHS=$EPOCHS"
COMMON_ENV+=" LR=$LR"
COMMON_ENV+=" GRAD_ACCUM=$GRAD_ACCUM"
COMMON_ENV+=" SEED=$SEED"
COMMON_ENV+=" DO_VALIDATE=1"
COMMON_ENV+=" EARLY_STOPPING=1"
COMMON_ENV+=" TEST_AFTER=1"
COMMON_ENV+=" AUTO_BS=1"
COMMON_ENV+=" USE_DDP=1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_train.sh"

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "Error: slurm_train.sh not found at $SLURM_SCRIPT"
    exit 1
fi

submit_job() {
    local job_name="$1"
    local model_name="$2"
    local extra_env="$3"
    local master_port="$4"

    local full_env="$COMMON_ENV $extra_env MODEL_NAME=$model_name MASTER_PORT=$master_port"
    local cmd="$full_env sbatch --job-name=$job_name $SLURM_SCRIPT"

    if [ "$DRY_RUN" == "1" ]; then
        echo "[DRY RUN] $cmd"
        echo ""
    else
        echo "Submitting: $job_name"
        echo "  Model name: $model_name"
        local result
        result=$(eval "$cmd")
        echo "  $result"
        echo ""
        echo "  Waiting 10s before next submission..."
        sleep 10
    fi
}

# ============================================
# Job 1: Combined (CheXpert-Plus + ReXGradient)
# ============================================
submit_job \
    "dino_${VIT_TAG}_${LORA_TAG}_combined" \
    "dinov3-${VIT_TAG}-${LORA_TAG}-combined" \
    "USE_MULTI=1" \
    12355

# ============================================
# Job 2: CheXpert-Plus only
# ============================================
submit_job \
    "dino_${VIT_TAG}_${LORA_TAG}_chexpert" \
    "dinov3-${VIT_TAG}-${LORA_TAG}-chexpert" \
    "USE_MULTI=0 CXR_FILEPATH=${CHEXPERT_DIR}/chexpert_plus_train.h5 TXT_FILEPATH=${CHEXPERT_DIR}/chexpert_plus_train_metadata.csv" \
    12356

# ============================================
# Job 3: ReXGradient only
# ============================================
submit_job \
    "dino_${VIT_TAG}_${LORA_TAG}_rexgrad" \
    "dinov3-${VIT_TAG}-${LORA_TAG}-rexgradient" \
    "USE_MULTI=0 CXR_FILEPATH=${REXGRADIENT_DIR}/rexgradient_train.h5 TXT_FILEPATH=${REXGRADIENT_DIR}/rexgradient_train_metadata.csv" \
    12357

# ============================================
# Summary
# ============================================
if [ "$DRY_RUN" == "1" ]; then
    echo "========================================================"
    echo "  Dry run complete. No jobs submitted."
    echo "========================================================"
else
    echo "========================================================"
    echo "  3 jobs submitted. Monitor with: squeue -u \$USER"
    echo ""
    echo "  Each job will:"
    echo "    - Auto-detect optimal batch size"
    echo "    - Validate periodically (CheXpert valid set)"
    echo "    - Save best model by mean AUC"
    echo "    - Early stop if no improvement"
    echo "    - Run final test (CheXpert test set) using best model"
    echo ""
    echo "  Results in: checkpoints/dinov3-${VIT_TAG}-${LORA_TAG}-{combined,chexpert,rexgradient}_*/test_results/"
    echo "========================================================"
fi
