#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-493  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 0-23:10:00
# Output files
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vagu0008@ad.umu.se

# Activate venv
srun -A NAISS2023-5-493 -n 10 -t 01:30:00 --gpus-per-node=A40:1 --pty bash
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis


# Activate the enviroment


module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-image/0.22.0
module load scikit-learn/1.3.1
module load h5py/3.9.0-foss-2023a

source PEFT_env/bin/activate
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit


model_name=$model_name
database=$database
paths=$paths
validation_strategy=$validation_strategy
experiment=$experiment
#!/usr/bin/bash
# RUN YOUR PROGRAM


python src/eval/classification/linear.py experiment/databases@db=AFC experiment/paths/system@_global_=alvis_dbg experiment/validation_strategy@_global_=loCo experiment=full experiment/models@_global_=medclip_resnet50


python src/eval/classification/linear.py experiment/databases@db=CoCross experiment/paths/system@_global_=alvis_dbg experiment/validation_strategy@_global_=5fold experiment=linear_probing_none_test_all experiment/models@_global_=medclip_vision

python src/eval/classification/linear.py experiment/databases@db=NY experiment/paths/system@_global_=alvis_dbg experiment/validation_strategy@_global_=5fold experiment=linear_probing_none_test_all experiment/models@_global_=medclip_vision



deactivate