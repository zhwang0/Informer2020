@echo off
REM Activate the Conda environment
call conda activate torch

python run_transformer_full_noise_rebuttal0_eval.py
python run_transformer_full_noise_rebuttal1_eval.py
python run_transformer_full_noise_rebuttal2_eval.py

conda deactivate

pause
