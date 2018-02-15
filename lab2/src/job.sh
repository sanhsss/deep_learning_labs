#!/bin/bash
$PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/common/software/anaconda3/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/common/cuda-8.0/lib64

srun python lab2_5.py --gpus 0

# Use 'wait' as a barrier to collect both executables when they are done.
wait