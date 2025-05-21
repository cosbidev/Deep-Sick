# SRUN MODE
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis

# Activate the enviroment





module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-image/0.22.0
module load scikit-learn/1.3.1
module load h5py/3.9.0-foss-2023a

source PEFT_env/bin/activate
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit


# RUN AGGREGATION

python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=AFC experiment/paths/system@_global_=alvis_2


python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=AFC_death experiment/paths/system@_global_=alvis_2

python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=CAR experiment/paths/system@_global_=alvis_2

python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=NY experiment/paths/system@_global_=alvis_2


python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=NY_icu experiment/paths/system@_global_=alvis_2

python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=CoCross experiment/paths/system@_global_=alvis_2

python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=NY_all experiment/paths/system@_global_=alvis_2

python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=NY_all_icu experiment/paths/system@_global_=alvis_2

# zip resul
cd /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results_BL/
zip -r /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results_BL/agg_all_BL_200225.zip aggregated_results
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit



# REMOVE ALL files .pth:
find /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results -name '*.pth' -delete
