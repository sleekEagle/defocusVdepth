#!/bin/bash
#SBATCH --job-name="midas"
#SBATCH --error="outputs/midas_50_2.err"
#SBATCH --output="outputs/midas_50_2.out"
#SBATCH --partition="gpu"
#SBATCH -w cheetah04
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
python train_def.py --max_lr 1e-4 --batch_size 1 --rgb_dir refocused_f_50_fdist_2 --data_path /p/blurdepth/data/ --depth_dir rawDepth --resultspth /p/blurdepth/models/defnet/ --blur_model midas --virtual_batch_size 12
