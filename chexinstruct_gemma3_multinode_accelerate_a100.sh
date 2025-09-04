#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH -N 2                          # number of nodes
#SBATCH --ntasks-per-node=1          # one task per GPU
#SBATCH --gpus-per-node=A40:4         # 4 GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH -t 0-02:00:00
#SBATCH -J "accelerate_test_trainer"
#SBATCH --error=trainer_TRAIN_%J.err
#SBATCH --output=trainer_TRAIN_%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it


set -euo pipefail


echo "=== Gemma3 Multi-Node Training (Direct SLURM Method) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"


# Show all available interfaces
echo "Available network interfaces:"
ip addr show | grep -E "^[0-9]+:" | awk '{print "  " $2}' | sed 's/://'

echo "=== Alvis Network Configuration ==="
echo "Node: $(hostname)"
echo "Available interfaces:"
ip addr show | grep -E "^[0-9]+:" | awk '{print "  " $2}' | sed 's/://'

export NETWORK_INTERFACE="ens27f0np0"

# Verify it exists and has IP
if ip addr show "$NETWORK_INTERFACE" 2>/dev/null | grep -q "inet "; then
    echo "✅ Interface $NETWORK_INTERFACE is configured and ready"
    INTERFACE_IP=$(ip addr show "$NETWORK_INTERFACE" | grep "inet " | head -1 | awk '{print $2}' | cut -d'/' -f1)
    echo "Interface IP: $INTERFACE_IP"
else
    echo "⚠️  $NETWORK_INTERFACE has no IP, checking VLAN interfaces..."

    # Try VLAN interfaces
    for vlan_if in ens27f0np0.1044 ens27f0np0.1043; do
        if ip addr show "$vlan_if" 2>/dev/null | grep -q "inet "; then
            NETWORK_INTERFACE="$vlan_if"
            echo "✅ Using VLAN interface: $NETWORK_INTERFACE"
            INTERFACE_IP=$(ip addr show "$NETWORK_INTERFACE" | grep "inet " | head -1 | awk '{print $2}' | cut -d'/' -f1)
            echo "Interface IP: $INTERFACE_IP"
            break
        fi
    done
fi


# Activate envc
source activateEnv.sh
echo "✓ Environment activated"

# Networking
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
export MASTER_ADDR MASTER_PORT NNODES WORLD_SIZE
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID


# NCCL
export NCCL_SOCKET_IFNAME="$NETWORK_INTERFACE"
export NCCL_IB_DISABLE=0
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Output dir
OUTPUT_DIR="./reports/finetune_gemma_findings_accelerate_zero3_debug_trainer"
mkdir -p "$OUTPUT_DIR"

# Training hyperparams
BATCH=8
EPOCHS=1
EVAL_STEPS=256
GRADIENT_ACCUMULATION_STEPS=1

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "NODE_RANK=$NODE_RANK"
echo "LOCAL_RANK=$LOCAL_RANK"

# Launch with srun (1 proc per GPU)
srun accelerate launch \
    --num_processes $WORLD_SIZE \
    --num_machines $SLURM_NNODES \
    --main_process_port $MASTER_PORT \
    --main_process_ip $MASTER_ADDR \
    src/finetune/finetune_accelerated_trainer.py \
    --deepspeed deepspeed/zero3.json \
    --lora_enable True  \
    --vision_lora False  \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_namespan_exclude lm_head embed_tokens \
    --num_lora_modules -1 \
    --dataset_name "chexinstruct" \
    --dataset_dir "data_chexinstruct/hf_parquet_gemma_format/gemma_3_findings" \
    --use_liger True \
    --model_name_or_path "google/gemma-3-4b-it" \
    --disable_flash_attn2 True \
    --freeze_projector False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --bf16 True \
    --fp16 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH \
    --per_device_eval_batch_size $BATCH \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate 1e-4 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --dataloader_num_workers 0 \
    --save_steps 1000 \
    --eval_steps $EVAL_STEPS \
    --save_total_limit 5 \
    --model_max_length 2048


echo "END TIME: $(date)"



