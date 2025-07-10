screen -S my_running_srun

srun --job-name=myjob --ntasks=1 --cpus-per-task=4  -A NAISS2024-5-577 --gpus-per-node=A40:2 --time=12:00:00 --pty bash
cd  /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick


module purge
module load Python/3.11.3-GCCcore-12.3.0
module load scikit-image/0.22.0
module load scikit-learn/1.3.1
module load h5py/3.9.0-foss-2023a


# Activate the enviroment
source Deep_Sick_env/bin/activate