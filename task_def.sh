#!/bin/bash
#SBATCH --job-name="40_3"
#SBATCH --error="40_3.err"
#SBATCH --output="40_3.out"
#SBATCH --partition="gpu"
#SBATCH -w cheetah04
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
python train_def.py --max_lr 1e-4 --batch_size 12 --rgb_dir refocused_f_40_fdist_3 --data_path /p/blurdepth/data/ --depth_dir rawDepth --resultspth /p/blurdepth/models/defnet/ --blur_model defnet
