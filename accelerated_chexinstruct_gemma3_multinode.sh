#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577
#SBATCH -p alvis
#SBATCH -N 4                                # number of nodes
#SBATCH --ntasks-per-node=4                # number of MP tasks per node
#SBATCH --gpus-per-node=A40:4              # number of GPUs per node
#SBATCH --cpus-per-task=160                # number of CPUs per task
#SBATCH -t 0-01:00:00
#SBATCH -J "accelerated_chexinstruct_gemma3_multinode"
#SBATCH --error=err_%J.err
#SBATCH --output=out_%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it

######################
### Set environment ###
######################
source activateEnvironment.sh
export GPUS_PER_NODE=4

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

######################
### Run Accelerate ###
######################

export WORKSPACE="/mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick"


accelerate launch \
    --config_file ${WORKSPACE}/deepspeed/ds_zero3_config_MultiNodes.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip ${head_node_ip} \
    --main_process_port 29800 \
    src/finetune/finetune_accelerated.py \
    --model_name_or_path google/gemma-3-4b-it \
    --dataset_dir data_chexinstruct/hf_parquet_gemma_format/gemma_findings \
    --output_dir ./reports/finetune_gemma_findings \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 10 \
    --with_tracking \
    --report_to wandb \
    --gradient_accumulation_steps 4 \
    --save_every_n_epochs 2 \
    --load_best_model

