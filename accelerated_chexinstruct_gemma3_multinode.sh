#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH -N 2                                # number of nodes
#SBATCH --ntasks-per-node=4                # One task per node (let accelerate handle GPU distribution)
#SBATCH --gpus-per-node=A100:4              # number of GPUs per node
#SBATCH --cpus-per-task=16                 # More CPUs per task since we have fewer tasks
#SBATCH -t 0-01:00:00
#SBATCH -J "accelerated_chexinstruct_gemma3_multinode"
#SBATCH --error=err_%J.err
#SBATCH --output=out_%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it

# Enable strict error handling
set -euo pipefail

echo "=== SLURM Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_NNODES"
echo "================================="

######################
### Set environment ###
######################
echo "=== Setting up environment ==="

# Move to workspace first
cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick || {
    echo "Error: Cannot change to workspace directory"
    exit 1
}

source activateEnv.sh
echo "✓ Environment activated"

export GPUS_PER_NODE=4
export WORKSPACE="/mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick"

######################
#### Set network #####
######################
echo "=== Setting up network ==="

# Use SLURM job ID to create unique port
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
unique_port=$((29000 + (SLURM_JOB_ID % 10000)))

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$unique_port
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=online

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

######################
### Verify setup #####
######################
echo "=== Verifying setup ==="

# Check required files
if [ ! -f "src/finetune/finetune_accelerated.py" ]; then
    echo "✗ Training script not found"
    exit 1
fi

if [ ! -f "deepspeed/ds_zero3_config_MultiNodes.yaml" ]; then
    echo "✗ DeepSpeed config not found"
    exit 1
fi

chmod +x src/finetune/finetune_accelerated.py
mkdir -p ./reports/finetune_gemma_findings

echo "✓ All checks passed"

######################
### Launch training ###
######################
echo "=== Launching training ==="

# Use a much simpler approach - let SLURM handle the distribution
srun accelerate launch \
    --config_file deepspeed/ds_zero3_config_MultiNodes.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    src/finetune/finetune_accelerated.py \
    --model_name_or_path google/gemma-3-4b-it \
    --dataset_dir data_chexinstruct/hf_parquet_gemma_format/gemma_3_findings \
    --output_dir ./reports/finetune_gemma_findings \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 8 \
    --with_tracking \
    --report_to wandb \
    --gradient_accumulation_steps 4 \
    --save_every_n_epochs 2 \
    --load_best_model

exit_code=$?
echo "Training completed with exit code: $exit_code"
exit $exit_code
