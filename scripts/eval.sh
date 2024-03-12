#!/bin/bash

#SBATCH --job-name=eval_brain-id
#SBATCH --gpus=1
#SBATCH --partition=

#SBATCH --mail-type=FAIL
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4	# 24
#SBATCH --mem=32G # 256G
#SBATCH --time=6-23:59:59
#SBATCH --output=logs/%j.log # Standard output and error log 


# exp-specific cfg #
exp_cfg_file='cfgs/eval/supv_bf.yaml'


date;hostname;pwd
python scripts/eval.py $exp_cfg_file 
date

