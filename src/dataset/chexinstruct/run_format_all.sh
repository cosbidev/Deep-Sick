#!/bin/bash

# Format ChexInstruct dataset for all supported models
# Usage: ./run_format_all.sh [input_dir] [base_output_dir]

set -e

# Configuration
INPUT_DIR=${1:-"data_chexinstruct/hf_parquet"}
BASE_OUTPUT_DIR=${2:-"data_chexinstruct"}
MODELS=("gemma" "llava15" "qwen25vl")

echo "🚀 Formatting ChexInstruct dataset for all models..."
echo "   Input: $INPUT_DIR"
echo "   Base output: $BASE_OUTPUT_DIR"
echo ""

# Format for each model
for model in "${MODELS[@]}"; do
    echo "📝 Processing model: $model"

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/hf_parquet_${model}_format"

    python src/dataset/chexinstruct/format_chexinstruct.py \
        --model "$model" \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR" \
        --verbose

    echo "✅ $model formatting complete!"
    echo ""
done

echo "🎉 All models formatted successfully!"

