#!/bin/bash

# CXR Concept Embedding Generation Script (GH200 node)
# Uses 3 LLM embedding models sequentially on a single GPU
# Target: ~492k cleaned concepts -> embedding vectors
# Models: SFR-Embedding-Mistral (7B), llama-embed-nemotron-8b (8B), KaLM-Embedding-Gemma3-12B (12B)
# Uses direct transformers inference (--embedding_type local) for GPU acceleration

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CSV_FILE="${1:-$SCRIPT_DIR/concepts_clean.csv}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/embeddings_output}"
BATCH_SIZE=32
MAX_LENGTH=4096
CONDA_ENV="ml311"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting CXR concept embedding generation on GH200..."
echo "Input file: $CSV_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE, Max length: $MAX_LENGTH"
echo "Conda env: $CONDA_ENV"
echo ""

# Function to run embedding generation
run_embedding() {
    local model_name="$1"
    local output_suffix="$2"

    echo "Starting $model_name ..."

    local resume_file="$OUTPUT_DIR/intermediate_${output_suffix}_indexed.pickle"
    local resume_flag=""
    if [ -f "$resume_file" ]; then
        echo "  Resuming from $resume_file"
        resume_flag="--resume $resume_file"
    fi

    micromamba run -n "$CONDA_ENV" python "$SCRIPT_DIR/get_embed.py" \
        --concepts_file "$CSV_FILE" \
        --embedding_type local \
        --local_model "$model_name" \
        --preserve_indices \
        --output "$OUTPUT_DIR/cxr_embeddings_${output_suffix}.pickle" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --device cuda \
        --use_fp16 \
        --trust_remote_code \
        --validate \
        --intermediate_file_path "$OUTPUT_DIR/intermediate_${output_suffix}.pickle" \
        $resume_flag

    echo "Completed $model_name"
    echo ""
}

# Run models sequentially (single GH200 GPU, models are 7B-12B)
# Model 1: SFR-Embedding-Mistral (7B, 4096-dim, last_token pooling)
run_embedding "Salesforce/SFR-Embedding-Mistral" "sfr_mistral"

# Model 2: llama-embed-nemotron-8b (8B, 4096-dim, average pooling)
run_embedding "nvidia/llama-embed-nemotron-8b" "nemotron_8b"

# Model 3: KaLM-Embedding-Gemma3-12B (12B, 3840-dim, lasttoken pooling)
run_embedding "tencent/KaLM-Embedding-Gemma3-12B-2511" "kalm_gemma3_12b"

echo ""
echo "All embedding generation completed!"
echo "Output files:"
echo "   - SFR-Mistral (4096-dim): $OUTPUT_DIR/cxr_embeddings_sfr_mistral.pickle"
echo "   - Nemotron-8B (4096-dim): $OUTPUT_DIR/cxr_embeddings_nemotron_8b.pickle"
echo "   - KaLM-Gemma3-12B (3840-dim): $OUTPUT_DIR/cxr_embeddings_kalm_gemma3_12b.pickle"
echo ""
echo "All embeddings maintain concept_idx -> embedding pairing"
echo "Intermediate files saved for resumption if needed"
