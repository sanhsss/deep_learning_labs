#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=300
#SBATCH -o results/output/job%Jo.txt
#SBATCH -e results/errors/job%Je.txt
# --exclude=node35,node118,node36
# 2880
# Use '&' to move the first job to the background

$PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/common/software/anaconda3/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/common/cuda-8.0/lib64

srun python lab7_var1.py --gpus 0

# Use 'wait' as a barrier to collect both executables when they are done.
wait