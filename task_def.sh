#!/bin/bash
#SBATCH --job-name="defocus"
#SBATCH --error="/p/blurdepth/results/defvdep/def.err"
#SBATCH --output="/p/blurdepth/results/defvdep/def.out"
#SBATCH --partition="gpu"
#SBATCH -w cheetah03
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
python train_def.py --max_lr 1e-4 --batch_size 4 --rgb_dir refocused_f_75_fdist_2 
