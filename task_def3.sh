#!/bin/bash
#SBATCH --job-name="depvdef"
#SBATCH --error="def.err"
#SBATCH --output="def.out"
#SBATCH --partition="gpu"
#SBATCH -w jaguar04
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
python train_def.py --max_lr 1e-4 --batch_size 12 --rgb_dir refocused_f_25_fdist_5 --data_path /p/blurdepth/data/ --depth_dir rawDepth --resultspth /p/blurdepth/models/defnet/
