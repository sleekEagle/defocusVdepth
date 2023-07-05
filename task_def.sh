#!/bin/bash
#SBATCH --job-name="depvdef"
#SBATCH --error="/p/blurdepth/models/defnet/def.err"
#SBATCH --output="/p/blurdepth/models/defnet/def.out"
#SBATCH --partition="gpu"
#SBATCH -w ristretto01
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
python train_def.py --max_lr 1e-4 --batch_size 4 --rgb_dir refocused_f_75_fdist_2 --data_path /p/blurdepth/data/ --depth_dir rawDepth --resultspth /p/blurdepth/models/defnet/