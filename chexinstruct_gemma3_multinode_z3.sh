#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH -N 4                        # 4 nodes
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gpus-per-node=A100:4       # 4 GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH -t 1-16:00:00
#SBATCH -J "gemma3_4MN_training"
#SBATCH --error=_TRAIN-FT_%J.err
#SBATCH --output=_TRAIN-FT_%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it
set -euo pipefail

# Gestione dei segnali per cleanup
cleanup() {
    echo "=== Cleanup started ==="
    # Termina tutti i processi Python
    pkill -f "finetune_accelerated_v2.py" || true
    # Pulisci la memoria GPU
    nvidia-smi --gpu-reset || true
    echo "=== Cleanup completed ==="
}
trap cleanup EXIT INT TERM

echo "=== Gemma3 Multi-Node Training (Direct SLURM Method) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"

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


# Move to project directory
cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick || { echo "Workspace not found"; exit 1; }

# Load environment
source activateEnv.sh
echo "✓ Environment activated"

# Configurazioni NCCL ottimizzate per stabilità
export WANDB_MODE=online
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_TREE_THRESHOLD=0  # Forza collective algorithms più stabili
export NCCL_P2P_DISABLE=0     # Abilita P2P per migliori performance
export NCCL_IB_TIMEOUT=22     # Aumenta timeout IB
export NCCL_BLOCKING_WAIT=1   # Usa blocking wait per stabilità
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Più dettagli per debug

# Configurazioni PyTorch per stabilità
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0  # Non bloccare per performance

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

export OUTPUT_DIR="./reports/finetune_gemma_findings_zero3_trainer_lora64"
mkdir -p "./reports/finetune_gemma_findings_zero3_trainer_lora_64"
mkdir -p "$OUTPUT_DIR"  # Assicurati che la directory di output esista


export BATCH=4
export EPOCHS=3
export EVAL_STEPS=64  # Riduci evaluation steps per testare più spesso
export GRADIENT_ACCUMULATION_STEPS=4  # Aumenta per compensare batch size ridotta

# Configurazioni HuggingFace
export TRANSFORMERS_VERBOSITY=warning  # Riduci verbosity
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Aggiungi timeout per processi bloccati
timeout 10800 srun bash -c '  # 3 ore di timeout
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
        --warmup_ratio 0.05 \
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
# Reset GPU se necessarios
nvidia-smi --gpu-reset || true

exit $exit_code