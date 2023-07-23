#!/bin/bash
#SBATCH --job-name="combined"
#SBATCH --error="combined.err"
#SBATCH --output="combined.out"
#SBATCH --partition="gpu"
#SBATCH -w jaguar04
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
python train_combined.py --max_lr 1e-4 --batch_size 8 --rgb_dir refocused_f_25_fdist_2 --data_path /p/blurdepth/data/ --depth_dir rawDepth --resultspth /p/blurdepth/models/defnet/ --blur_model defnet --geometry_model vpd
