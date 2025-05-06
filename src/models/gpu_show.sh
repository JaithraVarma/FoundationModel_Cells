#!/bin/bash

#SBATCH --job-name=CheckGPUs
#SBATCH --output=/nfs/tier2/users/sm1367/Cell_Model/src/models/gpu_check_%j.out
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=4G
#SBATCH --time=00:05:00
#SBATCH --nodelist=hslab-hcmp2

echo "Running on node: $SLURMD_NODENAME"
nvidia-smi