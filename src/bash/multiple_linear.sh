#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-493  -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 0-00:10:00
# Output files
#SBATCH -J "MULTIPLE_Linear_SBATCH"
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vagu0008@ad.umu.se

# Activate venv

cd /mimer/NOBACKUP/groups/naiss2023-6-336/ruffini/FM_PEFT_prognosis

# Activate the enviroment
source PEFT_env/bin/activate

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# Stampa degli argomenti di input

#!/usr/bin/env bashc


while getopts d:v:e: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        d) data=${OPTARG};;
        v) val=${OPTARG};;
        e) exp=${OPTARG};;

    esac
done
echo "data: $data";
echo "val: $val";

#!/usr/bin/bash
# RUN YOUR PROGRAM
python src/bash/launch_bash.py -d="$data" -e="$val" -c="$exp"
# Deactivate venv

deactivate
