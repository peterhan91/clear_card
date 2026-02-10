#!/bin/bash
#SBATCH --job-name=cxr_clip_train
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu

source /cbica/projects/CXR/miniconda3/etc/profile.d/conda.sh
conda activate chexzero

REPO_PATH="/cbica/projects/CXR/codes/clear_card"
cd "$REPO_PATH"

# ============================================
# Training Configuration
# ============================================
# TRAIN_SCRIPT: run_train (basic) or run_train_improved (DDP, validation, early stopping)
TRAIN_SCRIPT="${TRAIN_SCRIPT:-run_train_improved}"

# Preprocessed data root (each dataset in its own subdirectory)
DATA_ROOT="${DATA_ROOT:-/cbica/projects/CXR/data_p}"
CHEXPERT_DIR="${CHEXPERT_DIR:-${DATA_ROOT}/chexpert-plus}"
REXGRADIENT_DIR="${REXGRADIENT_DIR:-${DATA_ROOT}/rexgradient}"

# Single dataset (default: CheXpert-Plus train)
CXR_FILEPATH="${CXR_FILEPATH:-${CHEXPERT_DIR}/chexpert_plus_train.h5}"
TXT_FILEPATH="${TXT_FILEPATH:-${CHEXPERT_DIR}/chexpert_plus_train_metadata.csv}"

# Multi-dataset paths (CheXpert-Plus + ReXGradient)
# NOTE: Both metadata CSVs must have the same impression column name.
#   CheXpert uses 'impression' (lowercase), ReXGradient uses 'Impression' (uppercase).
#   To use multi-dataset training, first normalize column names:
#     python -c "import pandas as pd; df=pd.read_csv('${REXGRADIENT_DIR}/rexgradient_train_metadata.csv'); df.rename(columns={'Impression':'impression'}, inplace=True); df.to_csv('${REXGRADIENT_DIR}/rexgradient_train_metadata.csv', index=False)"
DATASET_PATHS="${DATASET_PATHS:-${CHEXPERT_DIR}/chexpert_plus_train.h5,${CHEXPERT_DIR}/chexpert_plus_train_metadata.csv ${REXGRADIENT_DIR}/rexgradient_train.h5,${REXGRADIENT_DIR}/rexgradient_train_metadata.csv}"

# Validation and test data
VAL_CXR="${VAL_CXR:-${CHEXPERT_DIR}/chexpert_plus_valid.h5}"
VAL_LABELS="${VAL_LABELS:-${REPO_PATH}/data/chexpert_valid.csv}"
TEST_CXR="${TEST_CXR:-${CHEXPERT_DIR}/chexpert_plus_test.h5}"
TEST_LABELS="${TEST_LABELS:-${REPO_PATH}/data/chexpert_test.csv}"

# ============================================
# Model Configuration
# ============================================
MODEL_NAME="${MODEL_NAME:-cxr-clip-v1.0}"
SAVE_DIR="${SAVE_DIR:-checkpoints}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-1e-4}"
SEED="${SEED:-1234}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

# DDP options (set USE_DDP=1 to enable 2-GPU distributed training)
USE_DDP="${USE_DDP:-1}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-12355}"

# DINOv3 options (set USE_DINOV3=1 to enable)
USE_DINOV3="${USE_DINOV3:-0}"
DINOV3_MODEL="${DINOV3_MODEL:-dinov3_vitb16}"
DINOV3_REPO_DIR="${DINOV3_REPO_DIR:-/cbica/projects/CXR/codes/dinov3}"
DINOV3_WEIGHTS_DIR="${DINOV3_WEIGHTS_DIR:-${DINOV3_REPO_DIR}/checkpoints}"
FREEZE_DINOV3="${FREEZE_DINOV3:-0}"

# Map model name to weights file
declare -A DINOV3_WEIGHT_MAP=(
    ["dinov3_vits16"]="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    ["dinov3_vits16plus"]="dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
    ["dinov3_vitb16"]="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    ["dinov3_vitl16"]="dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    ["dinov3_vith16plus"]="dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
    ["dinov3_vit7b16"]="dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
)
DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-${DINOV3_WEIGHTS_DIR}/${DINOV3_WEIGHT_MAP[$DINOV3_MODEL]}}"

# Multi-dataset training (set USE_MULTI=1 to enable)
USE_MULTI="${USE_MULTI:-0}"

# Validation during training (set DO_VALIDATE=1 to enable)
DO_VALIDATE="${DO_VALIDATE:-1}"

# Test after training (set TEST_AFTER=1 to enable)
TEST_AFTER="${TEST_AFTER:-1}"

# Early stopping (set EARLY_STOPPING=1 to enable)
EARLY_STOPPING="${EARLY_STOPPING:-1}"

echo "========================================"
echo "CXR CLIP Training Job"
echo "========================================"
echo "Train script:    $TRAIN_SCRIPT"
echo "Model name:      $MODEL_NAME"
echo "CXR filepath:    $CXR_FILEPATH"
echo "TXT filepath:    $TXT_FILEPATH"
echo "DDP:             $USE_DDP (${NUM_GPUS} GPUs)"
echo "Multi-dataset:   $USE_MULTI"
echo "Validation:      $DO_VALIDATE"
echo "DINOv3:          $USE_DINOV3 ($DINOV3_MODEL)"
echo "Batch size:      $BATCH_SIZE per GPU"
echo "Grad accum:      $GRAD_ACCUM"
echo "Effective batch: $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))"
echo "Epochs:          $EPOCHS"
echo "LR:              $LR"
echo "Save dir:        $SAVE_DIR"
echo "========================================"

# Build arguments
ARGS="--cxr_filepath $CXR_FILEPATH \
    --txt_filepath $TXT_FILEPATH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --seed $SEED \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --grad_accum_steps $GRAD_ACCUM"

if [ "$TRAIN_SCRIPT" == "run_train_improved" ]; then
    ARGS="$ARGS \
        --val_cxr_filepath $VAL_CXR \
        --val_label_path $VAL_LABELS \
        --chexpert_test_cxr $TEST_CXR \
        --chexpert_test_labels $TEST_LABELS"

    if [ "$USE_DDP" == "1" ]; then
        ARGS="$ARGS --use_ddp"
    fi

    if [ "$USE_MULTI" == "1" ]; then
        ARGS="$ARGS --use_multi_datasets --dataset_paths $DATASET_PATHS"
    fi

    if [ "$DO_VALIDATE" == "1" ]; then
        ARGS="$ARGS --do_validate"
    fi

    if [ "$TEST_AFTER" == "1" ]; then
        ARGS="$ARGS --test_after_training"
    fi

    if [ "$EARLY_STOPPING" == "1" ]; then
        ARGS="$ARGS --early_stopping"
    fi

    if [ "$USE_DINOV3" == "1" ]; then
        ARGS="$ARGS --use_dinov3 --dinov3_model_name $DINOV3_MODEL --dinov3_repo_dir $DINOV3_REPO_DIR --dinov3_weights $DINOV3_WEIGHTS"
        if [ "$FREEZE_DINOV3" == "1" ]; then
            ARGS="$ARGS --freeze_dinov3"
        fi
    fi

else
    # Basic run_train.py (single GPU only)
    ARGS="$ARGS --optimizer adam"
fi

# Launch: torchrun for DDP, python for single GPU
if [ "$USE_DDP" == "1" ] && [ "$TRAIN_SCRIPT" == "run_train_improved" ]; then
    CMD="torchrun --nproc-per-node=$NUM_GPUS --master-port=$MASTER_PORT training/${TRAIN_SCRIPT}.py $ARGS"
else
    CMD="python training/${TRAIN_SCRIPT}.py $ARGS"
fi

echo ""
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "========================================"
echo "Training complete!"
echo "Checkpoints saved to: $SAVE_DIR/$MODEL_NAME"
echo "========================================"
