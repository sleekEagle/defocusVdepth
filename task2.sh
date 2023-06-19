#!/bin/bash
#SBATCH --job-name="defocus"
#SBATCH --error="/p/blurdepth/results/defvdep/out3.err"
#SBATCH --output="/p/blurdepth/results/defvdep/out3.out"
#SBATCH --partition="gpu"
#SBATCH -w cheetah01
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
torchrun --nnodes=1 --nproc_per_node=4 train.py --max_lr 1e-6 --batch_size 4
