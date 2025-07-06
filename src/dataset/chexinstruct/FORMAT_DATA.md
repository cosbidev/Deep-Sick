# ChexInstruct Data Formatting Usage Guide

This guide explains how to format the ChexInstruct dataset for different multimodal models.

## File Structure
```
project/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── formatters.py  # Contains formatting functions
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── chexinstruct/
│   │       ├── __init__.py
│   │       ├── format_chexinstruct.py  # Main script for formatting
│   │       └── FORMAT_DATA.md  # This guide
│   └── data/
│       └── util_data.py   # Contains load_parquet_image_dataset, save_dataset_as_parquet
├── data_chexinstruct/
│   └── hf_parquet/        # Original dataset location
|
└── README.md
```


## Bash Runner Scripts

### 1. Simple Bash Script (run_format.sh)

```bash
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
python format_chexinstruct.py \
    --model "$MODEL" \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --verbose

echo ""
echo "✅ Formatting complete!"
```

### 2. Advanced Bash Script with Multiple Models (run_format_all.sh)

```bash
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
    
    python format_chexinstruct.py \
        --model "$model" \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR" \
        --verbose
    
    echo "✅ $model formatting complete!"
    echo ""
done

echo "🎉 All models formatted successfully!"
```

## Usage Examples

### 1. Format for Gemma
```bash
python format_chexinstruct.py --model gemma --input data_chexinstruct/hf_parquet --output data_chexinstruct/hf_parquet_gemma_format
```

### 2. Format for LLaVA 1.5
```bash
python format_chexinstruct.py --model llava15 --input data_chexinstruct/hf_parquet --output data_chexinstruct/hf_parquet_llava15_format
```

### 3. Format for Qwen2.5-VL
```bash
python format_chexinstruct.py --model qwen25vl --input data_chexinstruct/hf_parquet --output data_chexinstruct/hf_parquet_qwen25vl_format
```

### 4. Format specific splits only
```bash
python format_chexinstruct.py --model gemma --input data_chexinstruct/hf_parquet --output data_chexinstruct/hf_parquet_gemma_format --splits train val
```

### 5. Run with bash scripts
```bash
# Make scripts executable
chmod +x run_format.sh run_format_all.sh

# Format for single model
./run_format.sh gemma data_chexinstruct/hf_parquet data_chexinstruct/hf_parquet_gemma_format

# Format for all models
./run_format_all.sh data_chexinstruct/hf_parquet data_chexinstruct
```

## Expected Output Structure

After running the formatting scripts, you'll have:

```
data_chexinstruct/
├── hf_parquet/                    # Original dataset
├── hf_parquet_gemma_format/       # Gemma formatted
├── hf_parquet_llava15_format/     # LLaVA 1.5 formatted
└── hf_parquet_qwen25vl_format/    # Qwen2.5-VL formatted
```

## Troubleshooting

1. **Import Error**: Make sure the `src` directory is in your Python path
2. **File Not Found**: Verify the input directory exists and contains parquet files
3. **Permission Error**: Ensure you have write permissions for the output directory
4. **Memory Issues**: For large datasets, consider processing splits separately

## Requirements

- Python 3.8+
- torch
- datasets (HuggingFace)
- pandas
- pyarrow (for parquet support)