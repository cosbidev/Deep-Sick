#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577  -p alvis

#SBATCH -C NOGPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --gres=none
#SBATCH -J "Download CXR-LT"  # multi node, multi GPU
# Output files
#SBATCH --error=LINEAR_job_%J.err
#SBATCH --output=LINEAR_out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it
cd /mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick

wget -r -N -c -np  https://physionet.org/files/vindr-cxr/1.0.0/



