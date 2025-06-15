#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577  -p alvis
#SBATCH -C NOGPU
#SBATCH -N 1
#SBATCH -t 0-12:30:00
#SBATCH -J "Download R-G160k"  # multi node, multi GPU
# Output files
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it
cd  /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick


module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-image/0.22.0
module load scikit-learn/1.3.1

# Activate the enviroment
source Deep_Sick_env/bin/activate
python src/download/rex-gradient160k/downloadRexgradient160k.py


cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick/data/rex-gradient160k
# Concatenate the parts and extract the tar file
cat deid_png.part* > deid_png.tar
tar -xf deid_png.tar
