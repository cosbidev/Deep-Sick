#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100:4
#SBATCH -N 2
#SBATCH -t 0-00:20:00
#SBATCH -J "accelerated_chexinstruct_gemma3_1node_z2"
#SBATCH --error=err_%J.err
#SBATCH --output=out_%J.out
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


export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=online            # or "disabled" if you don't want to sync
export PYTHONPATH=${workspaceFolder}

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((10000 + ($RANDOM % 50000)))
export MASTER_ADDR=$head_node_ip
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

# Launch with Accelerate
accelerate launch \
    --mixed_precision bf16 \
    --config_file deepspeed/ds_zero2_config.yaml \
    --num_processes $GPUS_PER_NODE \
    --num_machines 1 \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
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
    --eval_steps 800 \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --remove_unused_columns false \
    --debug false \
