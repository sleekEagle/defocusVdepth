#!/bin/bash
#SBATCH --job-name="defocus"
#SBATCH --error="/p/blurdepth/results/defvdep/fr_encoder_fr_decoder.err"
#SBATCH --output="/p/blurdepth/results/defvdep/fr_encoder_fr_decoder.out"
#SBATCH --partition="gpu"
#SBATCH -w cheetah01
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
torchrun --nnodes=1 --nproc_per_node=4 train.py --max_lr 1e-3 --batch_size 4 --method 1 --freeze_encoder 1 --freeze_decoder 1
