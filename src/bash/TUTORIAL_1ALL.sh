#Per eseguire questo codice:
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis


module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-image/0.22.0
module load scikit-learn/1.3.1
module load h5py/3.9.0-foss-2023a

# Activate the enviroment
source PEFT_env/bin/activate


cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit





# ------------------------------------ EXTRACTOR ------------------------------------



bash src/bash/multiple_linear.sh -d AFC -v L -e FEXT

bash src/bash/multiple_linear.sh -d AFC_death -v L -e FEXT

bash src/bash/multiple_linear.sh -d CAR -v 5 -e FEXT

bash src/bash/multiple_linear.sh -d CoCross -v 5 -e FEXT

bash src/bash/multiple_linear.sh -d NY -v 5 -e FEXT

bash src/bash/multiple_linear.sh -d NY_icu -v 5 -e FEXT

bash src/bash/multiple_linear.sh -d NY_all -v 5 -e FEXT

bash src/bash/multiple_linear.sh -d NY_all_icu -v 5 -e FEXT


# ------------------------------------ VANILLA ------------------------------------
# AFC
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit


bash src/bash/multiple_linear.sh -d AFC -v L -e VANILLA (--)

bash src/bash/multiple_linear.sh -d AFC_death -v L -e VANILLA (--)

bash src/bash/multiple_linear.sh -d CAR -v 5 -e VANILLA (--)

bash src/bash/multiple_linear.sh -d CoCross -v 5 -e VANILLA (--)

bash src/bash/multiple_linear.sh -d NY -v 5 -e VANILLA (1812)

bash src/bash/multiple_linear.sh -d NY_icu -v 5 -e VANILLA (1812)

bash src/bash/multiple_linear.sh -d NY_all -v 5 -e VANILLA (--)

bash src/bash/multiple_linear.sh -d NY_all_icu -v 5 -e VANILLA (--)





# ------------------------------------ FSL ------------------------------------

bash src/bash/multiple_linear.sh -d AFC -v L -e FSL (v)

bash src/bash/multiple_linear.sh -d AFC_death -v L -e FSL (v)

bash src/bash/multiple_linear.sh -d CAR -v 5 -e FSL (v)

bash src/bash/multiple_linear.sh -d CoCross -v 5 -e FSL (v)

bash src/bash/multiple_linear.sh -d NY -v 5 -e FSL (v)

bash src/bash/multiple_linear.sh -d NY_icu -v 5 -e FSL  (v)


# ------------------------------------ FCSL ------------------------------------

bash src/bash/multiple_linear.sh -d AFC -v L -e FCSL

bash src/bash/multiple_linear.sh -d AFC_death -v L -e FCSL


# Aggregazione dei risultati

cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit



module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-image/0.22.0
module load scikit-learn/1.3.1
module load h5py/3.9.0-foss-2023a
source PEFT_env/bin/activate



python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=AFC
python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=AFC_death
python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=CAR
python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=NY
python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=NY_icu
python src/postprocess/aggregate_results/aggregate_results.py experiment/databases@db=CoCross

# zip results
cd /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results/
zip -r /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results/agg_24_02.zip aggregated_results
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis || exit

nvidia-smi --id=0


# FIND


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

cd /mimer/NOBACKUP/groups/snic2022-5-277/ruffini/FM_PEFT_prognosis/results
# Remove hold-out dirs
find  */CAR -type d -name 'hold-out' -print -exec rm -r {} +
find  */CoCross -type d -name 'hold-out' -print -exec rm -r {} +
find  */NY -type d -name 'hold-out' -print -exec rm -r {} +
find  */NY_icu -type d -name 'hold-out' -print -exec rm -r {} +

find  */AFC -type d -name 'hold-out' -print -exec rm -r {} +
find  */AFC_death -type d -name 'hold-out' -print -exec rm -r {} +


