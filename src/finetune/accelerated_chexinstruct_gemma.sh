#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100fat:4
#SBATCH -N 4
#SBATCH -t 7-00:00:00
#SBATCH -J "Linear Probing Training"
#SBATCH --error=LINEAR_job_%J.err
#SBATCH --output=LINEAR_out_%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it

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


export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=29700
export WANDB_MODE=online            # or "disabled" if you don't want to sync
export PYTHONPATH=${workspaceFolder}


# Launch with Accelerate
accelerate launch \
  --config_file deepspeed/ds_zero3_config.yaml \
  src/finetune/finetune_accelerated.py \
  --model_name_or_path google/gemma-3-4b-it \
  --dataset_name chexinstruct \
  --dataset_dir data_chexinstruct/hf_parquet_gemma_format/gemma_findings \
  --output_dir ./reports/finetune_gemma_findings \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 2 \
  --num_train_epochs 2 \
  --with_tracking \
  --report_to wandb


