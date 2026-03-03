#!/bin/bash
#SBATCH --job-name=cxr_zeroshot
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=zeroshot_%j.out
#SBATCH --error=zeroshot_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu

# Zero-shot concept-based classification on CheXpert test set
# Uses the best ViT-7B LoRA model (DINOv3) + 3 embedding models
#
# Requirements: ~80GB GPU memory for ViT-7B model
#   - If running on GH200, no SLURM needed; run directly.
#   - If running on A100 80GB, this script works.
#
# Usage:
#   sbatch slurm_zeroshot.sh                       # default: all 3 embedding models
#   MODELS="sfr_mistral" sbatch slurm_zeroshot.sh  # single model
#   MAX_CONCEPTS=10000 sbatch slurm_zeroshot.sh    # limit concepts for quick test

set -e

# Environment
CONDA_ENV="${CONDA_ENV:-ml311}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$CONDA_ENV"

REPO_PATH="/cbica/home/hanti/codes/clear_card"
cd "$REPO_PATH"

# Configuration
MODEL_PATH="${MODEL_PATH:-/cbica/projects/CXR/codes/clear_card/checkpoints/dinov3-vit7b16-lora-r32-a64-combined_vit7b16_20260224_144733/best_model.pt}"
MODELS="${MODELS:-sfr_mistral nemotron_8b kalm_gemma3_12b}"
MAX_CONCEPTS="${MAX_CONCEPTS:-0}"       # 0 = all ~492k concepts
IMAGE_BS="${IMAGE_BS:-16}"              # small for 7B model
CONCEPT_BS="${CONCEPT_BS:-512}"
SIM_BS="${SIM_BS:-64}"
BOOTSTRAP="${BOOTSTRAP:-0}"

echo "========================================"
echo "Zero-Shot Concept-Based Classification"
echo "========================================"
echo "Model:           ViT-7B LoRA (r=32, alpha=64)"
echo "Checkpoint:      $MODEL_PATH"
echo "Embedding models: $MODELS"
echo "Max concepts:    $MAX_CONCEPTS (0=all)"
echo "Image batch:     $IMAGE_BS"
echo "Concept batch:   $CONCEPT_BS"
echo "Bootstrap:       $BOOTSTRAP"
echo "Conda env:       $CONDA_ENV"
echo "========================================"

# Build command
ARGS="--model_path $MODEL_PATH \
    --embedding_models $MODELS \
    --max_concepts $MAX_CONCEPTS \
    --image_batch_size $IMAGE_BS \
    --concept_batch_size $CONCEPT_BS \
    --sim_batch_size $SIM_BS \
    --merge_lora"

if [ "$BOOTSTRAP" == "1" ]; then
    ARGS="$ARGS --bootstrap"
fi

CMD="python concepts/exp_zeroshot.py $ARGS"

echo ""
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "========================================"
echo "Zero-shot evaluation complete!"
echo "Results saved to: concepts/results/"
echo "========================================"
