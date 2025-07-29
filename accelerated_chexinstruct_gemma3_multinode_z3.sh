#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH -N 2                              # number of nodes
#SBATCH --ntasks-per-node=8              # One task per node (let accelerate handle GPU distribution)
#SBATCH --gpus-per-nodeA100:4           # number of GPUs per node
#SBATCH --cpus-per-task=4                # More CPUs per task since we have fewer tasks
#SBATCH -t 0-01:00:00
#SBATCH -J "accelerated_chexinstruct_gemma3_multinode_z3"
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
echo "=== Setting keep iup network ==="


# Use SLURM job ID to create unique port
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=online

# === Retry automatico sulla porta MASTER ===
head_node_ip=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_ADDR=$head_node_ip
for i in {1..5}; do
    export MASTER_PORT=$((10000 + RANDOM % 50000))
    echo "[INFO] Tentativo $i con MASTER_PORT=$MASTER_PORT"
    (exec 3<>/dev/tcp/$MASTER_ADDR/$MASTER_PORT) &>/dev/null && break || sleep 1
done

echo "$head_node_ip"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

######################
### Verify setup #####
######################
echo "=== Verifying setup ==="

# Check required files
if [ ! -f "src/finetune/finetune_accelerated_v2.py" ]; then
    echo "✗ Training script not found"
    exit 1
fi

if [ ! -f "deepspeed/ds_zero2_config.yaml" ]; then
    echo "✗ DeepSpeed config not found"
    exit 1
fi


chmod +x src/finetune/finetune_accelerated_v2.py
mkdir -p ./reports/finetune_gemma_findings_zero2_trainer_lora64

echo "✓ All checks passed"
######################
### Launch training ###
######################
echo "=== Launching training ==="

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=^lo,docker

# Use a much simpler approach - let SLURM handle the distribution
srun accelerate launch \
    --mixed_precision bf16 \
    --config_file deepspeed/ds_zero3_config.yaml \
    --num_processes $((SLURM_NNODES * $GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_NODEID \
    src/finetune/finetune_accelerated_v2.py \
    --model_name_or_path "google/gemma-3-4b-it" \
    --dataset_name "chexinstruct" \
    --dataset_dir "data_chexinstruct/hf_parquet_gemma_format/gemma_3_findings" \
    --output_dir "./reports/finetune_gemma_findings_zero2_trainer_lora64" \
    --learning_rate "2e-4" \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --num_train_epochs 3 \
    --report_to "wandb" \
    --preprocessing_num_workers 1 \
    --weight_decay "0.001" \
    --warmup_ratio "0.01" \
    --model_max_length 1500 \
    --lora_enable true \
    --lora_alpha 64 \
    --lora_r 64 \
    --gradient_checkpointing true \
    --peft_strategy "lora_gaussian" \
    --gradient_accumulation_steps 8 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --logging_steps 10 \
    --save_total_limit 3 \
    --load_best_model_at_end false \
    --eval_steps 25 \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --remove_unused_columns false \
    --verbose_logging true


exit_code=$?
echo "Training completed with exit code: $exit_code"
exit $exit_code

