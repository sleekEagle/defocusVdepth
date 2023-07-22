#!/bin/bash
#SBATCH --job-name="vpd"
#SBATCH --error="vpd.err"
#SBATCH --output="vpd.out"
#SBATCH --partition="gpu"
#SBATCH -w jaguar06
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
torchrun --nnodes=1 --nproc_per_node=2 train_VPD.py --max_lr 1e-5 --batch_size 4  --freeze_encoder 1  --rgb_dir refocused_f_25_fdist_2 --data_path /p/blurdepth/data/ --depth_dir rawDepth --resultspth /p/blurdepth/models/defnet/ --model_name vpd --resume_from /p/blurdepth/models/vpd_depth_480x480.pth
