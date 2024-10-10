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

module load cuda/gcc/9.4.0/zen2/11.6.2 cudnn
module load pytorch

python run_informer_prob_noise_rebuttal0.py