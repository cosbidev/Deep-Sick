#!/usr/bin/env bash



# Srun in local DEBUG:
srun -A NAISS2023-5-493 -n 10 -t 02:30:00 --gpus-per-node=A40:1 --pty bash

#Per eseguire questo codice:
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis


#Muliple run
bash src/bash/multiple_linear.sh -d NY -v 5 -e VANILLA
bash src/bash/multiple_linear.sh -d NY -v 5 -e FSL

bash src/bash/multiple_linear.sh -d NY_icu -v 5 -e VANILLA
bash src/bash/multiple_linear.sh -d NY_icu -v 5 -e FSL







# SRUN MODE
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis

# Activate the enviroment
source PEFT_env/bin/activate



module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-image/0.22.0
module load scikit-learn/1.3.1
module load h5py/3.9.0-foss-2023a


cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit



# DEBUG MODE:
python src/eval/classification/linear.py experiment/databases@db=NY experiment/paths/system@_global_=alvis experiment/validation_strategy@_global_=5fold experiment=linear_probing_none_test_all experiment/models@_global_=resnet_18

# DEBUG MODE:
WANDB__SERVICE_WAIT=300 python src/eval/classification/linear.py experiment/databases@db=AFC experiment/paths/system@_global_=alvis experiment/validation_strategy@_global_=loCo experiment=linear_probing_LoRA experiment/models@_global_=biomedclip



# Post PROCESS
python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=NY experiment/paths/system@_global_=alvis



# zip results
cd /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results/
zip -r /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results/agg.zip aggregated_results
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit



# REMOVE ALL files:
find /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results/linear_probing_test_all_few_shot_learning_LoRa_8 -name '*.pth' -delete
