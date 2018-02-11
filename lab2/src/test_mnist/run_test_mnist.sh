#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=30
#SBATCH -o results/output/job%Jo.txt
#SBATCH -e results/errors/job%Je.txt
# --exclude=node35,node118,node36
# 2880
# Use '&' to move the first job to the background

# export PATH=$PATH:/common/software/anaconda3/lib
#export PATH=/common/home/deeplearning/studdocs/gerasimov_d/LAB2/mxnet-cuda/mxnet_cu80-0.11.0.dist-info:$PATH
#export PYTHONPATH=/common/home/deeplearning/studdocs/gerasimov_d/LAB2/mxnet-cuda/mxnet_cu80-0.11.0.dist-info:$PYTHONPATH
#export PATH=/common/home/deeplearning/studdocs/gerasimov_d/LAB2/mxnet-cuda/mxnet:$PATH
#export PYTHONPATH=/common/home/deeplearning/studdocs/gerasimov_d/LAB2/mxnet-cuda/mxnet:$PYTHONPATH

$PATH
#$PYTHONPATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/common/software/anaconda3/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/common/cuda-8.0/lib64

srun python mnist.py --gpus 0

# Use 'wait' as a barrier to collect both executables when they are done.
wait
