#!/bin/bash

#SBATCH --job-name=su-se
#SBATCH --gpus=1
#SBATCH --partition=rtx8000 # dgx-a100, rtx6000, rtx8000, lcnrtx, lcnv100

#SBATCH --mail-type=FAIL
#SBATCH --account=lcnrtx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4	# 24
#SBATCH --mem=32G # 256G
#SBATCH --time=6-23:59:59
#SBATCH --output=/autofs/space/yogurt_002/users/pl629/logs/%j.log # Standard output and error log 


# exp-specific cfg #
exp_cfg_file=/autofs/space/yogurt_003/users/pl629/code/BrainID/cfgs/train/supv_seg.yaml


date;hostname;pwd
python /autofs/space/yogurt_003/users/pl629/code/BrainID/scripts/train.py $exp_cfg_file 
date
