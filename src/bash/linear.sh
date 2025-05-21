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

# Activate venv



cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis


module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-image/0.22.0
module load scikit-learn/1.3.1
module load h5py/3.9.0-foss-2023a


# Activate the enviroment
source PEFT_env/bin/activate


cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit

model_name=$model_name
database=$database
paths=$paths
validation_strategy=$validation_strategy
experiment=$experiment
echo "model_name: $model_name";
echo "database: $database";
echo "paths: $paths";
echo "validation_strategy: $validation_strategy";


#!/usr/bin/bash
# RUN YOUR PROGRAM
WANDB__SERVICE_WAIT=300 python src/eval/classification/linear.py experiment/databases@db="$database" experiment/paths/system@_global_=alvis experiment/validation_strategy@_global_="$validation_strategy" experiment="$experiment" experiment/models@_global_="$model_name"
export WANDB_MODE=offline

cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit
# REMOVE ALL files:

deactivate



