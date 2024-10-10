@echo off
REM Activate the Conda environment
call conda activate torch

python run_informer_prob_noise_rebuttal1_eval.py
python run_informer_prob_noise_rebuttal2_eval.py

conda deactivate

pause
