#!/bin/bash

#SBATCH --job-name=DINO_Train
#SBATCH --output=/nfs/tier2/users/sm1367/Cell_Model/src/models/dino_%j.out
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=1-00:00:00
#SBATCH --nodelist=hslab-hcmp1

# Activate conda environment
source /nfs/tier2/users/sm1367/anaconda3/bin/activate
conda activate cellmodel

# Set CUDA device to GPU 2
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run the script with save_interval (from prior fix)
python /nfs/tier2/users/sm1367/Cell_Model/src/models/dino_new2_2504_train.py --test_num 6

echo "DONE"