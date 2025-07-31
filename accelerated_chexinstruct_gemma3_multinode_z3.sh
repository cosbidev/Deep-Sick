#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH -N 4                         # two nodes
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gpus-per-node=A100:4       # 4 GPUs per node
#SBATCH --cpus-per-task=4
#SBATCH -t 1-12:00:00
#SBATCH -J "gemma3_MN_training_z3"
#SBATCH --error=training_%J.err
#SBATCH --output=training_%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it
set -euo pipefail

echo "=== Gemma3 Multi-Node Training (Direct SLURM Method) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"

# Setup networking (same as diagnostic)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((30000 + RANDOM % 10000))

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$SLURM_NTASKS"


# Move to project directory
cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick || { echo "Workspace not found"; exit 1; }

# Load environment
source activateEnv.sh
echo "✓ Environment activated"


export WANDB_MODE=online


# NCCL settings (same as diagnostic)
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN  # Reduce noise

# Disable distributed debug to reduce noise
export TORCH_DISTRIBUTED_DEBUG=INFO

######################
### Verify Training Files
######################
[[ -f src/finetune/finetune_accelerated_v2.py ]] || { echo "Training script missing"; exit 1; }
[[ -f deepspeed/ds_zero3_config.yaml ]] || { echo "DeepSpeed config missing"; exit 1; }

######################
### Launch Training (EXACT SAME METHOD AS DIAGNOSTIC)
######################

echo "=== Launching Training with Direct SLURM (Working Method) ==="
# PARAMS for the training script
export ACCELERATE_CONFIG_FILE="deepspeed/ds_zero3_config.yaml"
# Use the EXACT same srun pattern that worked in diagnostics

export OUTPUT_DIR="./reports/finetune_gemma_findings_zero3_trainer_lora64"
mkdir -p ./reports/finetune_gemma_findings_zero3_trainer_lora64
export BATCH=4  # Adjust batch size as needed
export EPOCHS=3  # Adjust number of epochs as needed
export EVAL_STEPS=128  # Adjust evaluation steps as needed
export GRADIENT_ACCUMULATION_STEPS=4  # Adjust gradient accumulation steps as needed

srun bash -c '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export WORLD_SIZE=$SLURM_NTASKS
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'

  echo "[Rank $RANK] Starting training on $(hostname) with LOCAL_RANK=$LOCAL_RANK"
  # Run the training script directly (no accelerate launcher)
    src/finetune/finetune_accelerated_v2.py \
        --deepspeed_config_file '"$ACCELERATE_CONFIG_FILE"' \
        --model_name_or_path "google/gemma-3-4b-it" \
        --dataset_name "chexinstruct" \
        --dataset_dir "data_chexinstruct/hf_parquet_gemma_format/gemma_3_findings" \
        --output_dir '"$OUTPUT_DIR"' \
        --learning_rate 2e-4 \
        --lr_scheduler_type "cosine_with_restarts" \
        --per_device_train_batch_size '"$BATCH"' \
        --per_device_eval_batch_size '"$BATCH"' \
        --num_train_epochs '"$EPOCHS"'\
        --report_to wandb \
        --preprocessing_num_workers 1 \
        --weight_decay 0.0001 \
        --warmup_ratio 0.03 \
        --model_max_length 1500 \
        --lora_enable true \
        --lora_alpha 64 \
        --lora_r 64 \
        --gradient_checkpointing true \
        --peft_strategy "lora_gaussian" \
        --gradient_accumulation_steps '"$GRADIENT_ACCUMULATION_STEPS"' \
        --eval_steps '"$EVAL_STEPS"' \
        --checkpointing_strategy epoch \
        --checkpointing_divider 1 \
        --load_best_model true \
        --verbose_logging true \
        --bf16 true
'


exit_code=$?
echo "Training completed with exit code: $exit_code"
exit $exit_code


