#!/bin/bash

# ChexInstruct Dataset Formatter Runner
# Usage: ./run_format.sh [model] [input_dir] [output_dir]

set -e  # Exit on any error

# Default values
MODEL=${1:-"gemma"}
INPUT_DIR=${2:-"data_chexinstruct/hf_parquet"}
OUTPUT_DIR=${3:-"data_chexinstruct/hf_parquet_${MODEL}_format"}

echo "🚀 Starting ChexInstruct dataset formatting..."
echo "   Model: $MODEL"
echo "   Input: $INPUT_DIR"
echo "   Output: $OUTPUT_DIR"
echo ""

# Run the formatting script
python src/dataset/chexinstruct/format_chexinstruct.py \
    --model "$MODEL" \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --verbose

echo ""
echo "✅ Formatting complete!"
