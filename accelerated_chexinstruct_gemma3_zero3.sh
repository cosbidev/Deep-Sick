#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100fat:4
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH -J "accelerated_chexinstruct_gemma3"
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



# Launch with Accelerate
accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port 29800 \
    --config_file deepspeed/ds_zero3_config.yaml \
    src/finetune/finetune_accelerated_v2.py \
    --model_name_or_path "google/gemma-3-4b-it" \
    --dataset_name "chexinstruct" \
    --dataset_dir "data_chexinstruct/hf_parquet_gemma_format/gemma_3_findings" \
    --output_dir "./reports/finetune_gemma_findings_zero3_trainer" \
    --learning_rate "2e-4" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 2 \
    --report_to "wandb" \
    --preprocessing_num_workers 1 \
    --weight_decay "0.0" \
    --warmup_ratio "0.03" \
    --model_max_length 1500 \
    --gradient_checkpointing false \
    --lora_enable true \
    --lora_alpha 64 \
    --lora_r 64 \
    --peft_strategy "lora_gaussian" \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --logging_steps 10 \
    --save_total_limit 3 \
    --load_best_model_at_end false \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --remove_unused_columns false \
    --verbose_logging true