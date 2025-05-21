#!/usr/bin/env bash



# Srun in local DEBUG:
srun -A NAISS2023-5-493 -n 10 -t 02:30:00 --gpus-per-node=A40:1 --pty bash

#Per eseguire questo codice:
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis


#Muliple run

bash src/bash/multiple_linear.sh -d AFC -v L -e VANILLA
bash src/bash/multiple_linear.sh -d AFC -v L -e FSL
bash src/bash/multiple_linear.sh -d AFC -v L -e FCSL



bash src/bash/multiple_linear.sh -d AFC_death -v L -e VANILLA
bash src/bash/multiple_linear.sh -d AFC_death -v L -e FSL
bash src/bash/multiple_linear.sh -d AFC_death -v L -e FCSL





# SRUN MODE
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis

# Activate the enviroment
source PEFT_env/bin/activate


cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit


module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-image/0.22.0
module load scikit-learn/1.3.1
module load h5py/3.9.0-foss-2023a
source PEFT_env/bin/activate




# DEBUG MODE:
python src/eval/classification/linear.py experiment/databases@db=AFC experiment/paths/system@_global_=alvis experiment/validation_strategy@_global_=loCo experiment=linear_probing experiment/models@_global_=resnet_50

# DEBUG MODE:
WANDB__SERVICE_WAIT=300 python src/eval/classification/linear.py experiment/databases@db=AFC experiment/paths/system@_global_=alvis experiment/validation_strategy@_global_=loCo experiment=linear_probing_LoRA experiment/models@_global_=biomedclip



# Post PROCESS
python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=AFC experiment/paths/system@_global_=alvis



# zip results
cd /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results/
zip -r /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results/agg.zip aggregated_results
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit



# REMOVE ALL files:
find /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results -name '*.pth' -delete
# Remove dir with a certain name

find /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results -type d -name 'AFC_death_mortality*' -print -exec rm -r {} +


#Per eseguire questo codice:
find  */CAR -type d -name 'loCo' -print
find  */CoCross -type d -name 'loCo' -print
find  */NY -type d -name 'loCo' -print
find  */NY_icu -type d -name 'loCo' -print

find  */AFC -type d -name '5fold' -print
find  */AFC_death -type d -name '5fold' -print


find  */CAR -type d -name 'hold-out' -print
find  */CoCross -type d -name 'hold-out' -print
find  */NY -type d -name 'hold-out' -print
find  */NY_icu -type d -name 'hold-out' -print

find  */AFC -type d -name 'hold-out' -print
find  */AFC_death -type d -name 'hold-out' -print



# Remove dir with a certain name
find  */CAR -type d -name 'loCo' -print -print -exec rm -r {} +
find  */CoCross -type d -name 'loCo' -print -print -exec rm -r {} +
find  */NY -type d -name 'loCo' -print  -print -exec rm -r {} +
find  */NY_icu -type d -name 'loCo' -print -print -exec rm -r {} +
find  */AFC -type d -name '5fold' -print -print -exec rm -r {} +
find  */AFC_death -type d -name '5fold' -print -print -exec rm -r {} +


find  */CAR -type d -name 'hold-out' -print -exec rm -r {} +
find  */CoCross -type d -name 'hold-out' -print -exec rm -r {} +
find  */NY -type d -name 'hold-out' -print -exec rm -r {} +
find  */NY_icu -type d -name 'hold-out' -print -exec rm -r {} +

find  */AFC -type d -name 'hold-out' -print -exec rm -r {} +
find  */AFC_death -type d -name 'hold-out' -print -exec rm -r {} +