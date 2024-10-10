#!/bin/bash 
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=102400
#SBATCH --gpus=a100:1
#SBATCH -t 12:00:0

#SBATCH --mail-user=zhwang1@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load cuda cudnn
module load pytorch

source ~/my-venv/bin/activate

python run_informer_full_noise.py