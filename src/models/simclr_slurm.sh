#!/bin/bash

#SBATCH --job-name=SimCLR_Train
#SBATCH --output=/nfs/tier2/users/sm1367/Cell_Model/src/models/simclr_%j.out
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=1-00:00:00
#SBATCH --nodelist=hslab-hcmp1

# Activate conda environment
source /nfs/tier2/users/sm1367/anaconda3/bin/activate
conda activate cellmodel

# Set CUDA device to GPU 0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run the script
python /nfs/tier2/users/sm1367/Cell_Model/src/models/simclr_train_model.py

echo "DONE"