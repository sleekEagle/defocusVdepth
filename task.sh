#!/bin/bash
#SBATCH --job-name="defocus"
#SBATCH --error="/p/blurdepth/results/defvdep/out.err"
#SBATCH --output="/p/blurdepth/results/defvdep/out.out"
#SBATCH --partition="gpu"
#SBATCH -w cheetah04
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
torchrun --nnodes=1 --nproc_per_node=4 train.py --max_lr 1e-4 --batch_size 4 --method 0 --freeze_encoder 0
