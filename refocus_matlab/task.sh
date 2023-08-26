#!/bin/bash
#SBATCH --job-name="refocus50_2"
#SBATCH --error="out.err"
#SBATCH --output="out.out"
#SBATCH -w cortado01
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
echo "running..."
module load matlab
echo "loaded module"
matlab -nodisplay -nosplash -nodesktop -r "run('blur_images.m');exit;"
echo "done"
