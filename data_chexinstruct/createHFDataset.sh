#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577  -p alvis

#SBATCH -C NOGPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --gres=none
#SBATCH -J "Download CXR-LT"  # multi node, multi GPU
# Output files
#SBATCH --error=LINEAR_job_%J.err
#SBATCH --output=LINEAR_out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it
cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load scikit-image/0.22.0
module load scikit-learn/1.3.1
module load h5py/3.9.0-foss-2023a


# Activate the enviroment
source Deep_Sick_env/bin/activate

python src/dataset/chexinstruct/createHFDataset.py