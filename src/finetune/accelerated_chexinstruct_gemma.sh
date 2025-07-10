#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577  -p alvis

#SBATCH --gpus-per-node=A40:1
#SBATCH -N 2
#SBATCH -t 2-12:30:00
#SBATCH -J "Linear Probing Training"  # multi node, multi GPU
# Output files
#SBATCH --error=LINEAR_job_%J.err
#SBATCH --output=LINEAR_out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it

cd  /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick



module purge
module load Python/3.11.3-GCCcore-12.3.0
module load scikit-image/0.22.0
module load scikit-learn/1.3.1
module load h5py/3.9.0-foss-2023a


# Activate the enviroment
source Deep_Sick_env/bin/activate


accelerate launch --config_file {file} src/finetune/finetune_accelerated.py \


--dataset_name "chexinstruct" \
--model_name_or_path "google/gemma-3-4b-it" \
--dataset_dir "data_chexinstruct/hf_parquet_gemma_format/gemma_findings" \
--block_size 2048 \
--output_dir "./reports/finetune_gemma_findings" \
--learning_rate 2e-4 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 24 \
--num_train_epochs 30 \
--with_tracking \
--report_to "wandb"\