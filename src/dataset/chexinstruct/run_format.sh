#!/bin/bash

# Move to project directory
cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick

# Load modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load scikit-image/0.22.0
module load scikit-learn/1.3.1
module load h5py/3.9.0-foss-2023a
module load CUDA/12.1.1

# Activate the virtual environment
source Deep_Sick_env/bin/activate


export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=online            # or "disabled" if you don't want to sync
export PYTHONPATH=${workspaceFolder}

# Run the formatting script
python src/dataset/chexinstruct/format_chexinstruct.py \
                --model_name_or_path google/gemma-3-4b-it \
                --dataset_name chexinstruct \
                --dataset_dir data_chexinstruct/hf_parquet_gemma_format/gemma_findings \
                --preprocessing_num_workers 8

