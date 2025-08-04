#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH -N 2                         # 4 nodes
#SBATCH --ntasks-per-node=1         # one task per GPU
#SBATCH --gpus-per-node=A100:4       # 4 GPUs per node (A40)
#SBATCH --cpus-per-task=16
#SBATCH -t 0-00:45:00
#SBATCH -J "gemma3_MN_training_z3"
#SBATCH --error=training_z3_%J.err
#SBATCH --output=training_z3_%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it
set -euo pipefail

echo "=== Gemma3 Multi-Node Training (Direct SLURM Method) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"

# Setup networking (same as diagnostic)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


MASTER_PORT=$((30000 + RANDOM % 10000))
echo "GPU ON NODE = $SLURM_GPUS_ON_NODE"
export GPUs_PER_NODE=4

export WORLD_SIZE=$((SLURM_NNODES * $GPUs_PER_NODE))
export RANK=$SLURM_PROCID




echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"



# Move to project directory
cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick || { echo "Workspace not found"; exit 1; }

# Load environment
source activateEnv.sh
echo "✓ Environment activated"


export WANDB_MODE=online



export NCCL_PROTO=simple

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

#export NCCL_ALGO=ring
export NCCL_DEBUG=info
#export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

# NCCL settings (same as diagnostic)
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
export ACCELERATE_USE_SLURM=true

export OUTPUT_DIR="./reports/finetune_gemma_findings_zero3_trainer_lora64"
mkdir -p ./reports/finetune_gemma_findings_zero3_trainer_lora64
export BATCH=4  # Adjust batch size as needed
export EPOCHS=3  # Adjust number of epochs as needed
export EVAL_STEPS=64  # Adjust evaluation steps as needed
export GRADIENT_ACCUMULATION_STEPS=4  # Adjust gradient accumulation steps as needed


######################
######################
#### Set network #####
######################
######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * $GPUs_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    "
export SCRIPT="src/finetune/finetune_accelerated_v2.py"
export SCRIPT_ARGS=" \
    --deepspeed_config_file $ACCELERATE_CONFIG_FILE \
    --model_name_or_path google/gemma-3-4b-it \
    --dataset_name chexinstruct \
    --dataset_dir data_chexinstruct/hf_parquet_gemma_format/gemma_3_findings \
    --output_dir $OUTPUT_DIR \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine_with_restarts \
    --per_device_train_batch_size $BATCH \
    --per_device_eval_batch_size $BATCH \
    --num_train_epochs $EPOCHS\
    --report_to wandb \
    --preprocessing_num_workers 1 \
    --weight_decay 0.0001 \
    --warmup_ratio 0.05 \
    --model_max_length 1500 \
    --lora_enable true \
    --lora_alpha 64 \
    --lora_r 64 \
    --gradient_checkpointing true \
    --peft_strategy lora_gaussian \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --eval_steps $EVAL_STEPS \
    --checkpointing_strategy epoch \
    --checkpointing_divider 1 \
    --load_best_model true \
    --verbose_logging true \
    --bf16 true \
    "
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
srun $CMD



