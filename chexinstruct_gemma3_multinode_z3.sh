#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH -N 4                        # two nodes
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gpus-per-node=A40:4       # 4 GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH -t 0-06:00:00               # Aumenta il timeout
#SBATCH -J "gemma3_MN_TRAIN_lora_vanilla"
#SBATCH --error=_TRAIN_%J.err
#SBATCH --output=_TRAIN_%J.out
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

# =============================================================================
# NETWORK ENVIRONMENT CONFIGURATION
# =============================================================================
# Move to project directory
cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick || { echo "Workspace not found"; exit 1; }

# Load environment
source activateEnv.sh
echo "✓ Environment activated"
# Setup networking (same as diagnostic)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((30000 + RANDOM % 10000))


NODES=$SLURM_NNODES
TASKS_PER_NODE=$SLURM_NTASKS_PER_NODE
WORLD_SIZE=$((NODES * TASKS_PER_NODE))


echo "GPU ON NODE = $SLURM_GPUS_ON_NODE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
# Configure NCCL with detected interface
export NCCL_SOCKET_IFNAME="$NETWORK_INTERFACE"
export NCCL_IB_DISABLE=0
# TORCH
export TORCH_DISTRIBUTED_DEBUG=INFO
# =============================================================================

# Move to project directory
cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick || { echo "Workspace not found"; exit 1; }

# Load environment
source activateEnv.sh
echo "✓ Environment activated"

# Disable distributed debug to reduce noise


export WANDB_MODE=online
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




export OUTPUT_DIR="./reports/finetune_gemma_findings_zero3_trainer_lora64_vanilla"

mkdir -p "$OUTPUT_DIR"  # Assicurati che la directory di output esista


export BATCH=4
export EPOCHS=4
export EVAL_STEPS=64  # Riduci evaluation steps per testare più spesso
export GRADIENT_ACCUMULATION_STEPS=8  # Aumenta per compensare batch size ridotta

# Aggiungi timeout per processi bloccati
srun bash -c '  # 3 ore di timeout
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export WORLD_SIZE='"$WORLD_SIZE"'
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'

  echo "[Rank $RANK] Starting training on $(hostname) with LOCAL_RANK=$LOCAL_RANK"
  echo "WORLD_SIZE = $WORLD_SIZE"
  echo "OUTPUT_DIR = '"$OUTPUT_DIR"'"

  # Crea directory di output per questo processo
  mkdir -p '"$OUTPUT_DIR"'

  # Run the training script directly (no accelerate launcher)
  python src/finetune/finetune_accelerated_v2.py \
        --deepspeed_config_file '"$ACCELERATE_CONFIG_FILE"' \
        --model_name_or_path "google/gemma-3-4b-it" \
        --dataset_name "chexinstruct" \
        --dataset_dir "data_chexinstruct/hf_parquet_gemma_format/gemma_3_findings" \
        --output_dir '"$OUTPUT_DIR"' \
        --learning_rate 2e-4 \
        --lr_scheduler_type "cosine_with_restarts" \
        --per_device_train_batch_size '"$BATCH"' \
        --per_device_eval_batch_size '"$BATCH"' \
        --num_train_epochs '"$EPOCHS"' \
        --report_to wandb \
        --preprocessing_num_workers 1 \
        --weight_decay 0.0001 \
        --warmup_ratio 0.01 \
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
        --verbose_logging false \
        --bf16 true \
        --debug false
'
exit_code=$?
echo "Training completed with exit code: $exit_code"

# Cleanup finale
echo "=== Final cleanup ==="
# Termina processi rimasti
pkill -f "python.*finetune" || true
# Reset GPU se necessario
nvidia-smi --gpu-reset || true
exit $exit_code
