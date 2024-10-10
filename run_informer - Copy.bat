@echo off
REM Activate the Conda environment
call conda activate torch

python run_informer_prob_noise_rebuttal3_eval.py
python run_informer_prob_noise_rebuttal4_eval.py

conda deactivate

pause
