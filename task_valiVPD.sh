#!/bin/bash
#SBATCH --job-name="vpd"
#SBATCH --error="vpd.err"
#SBATCH --output="vpd.out"
#SBATCH --partition="gpu"
#SBATCH -w jaguar06
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
python eval_VPD.py --rgb_dir refocused_f_25_fdist_1 --data_path /p/blurdepth/data/ --depth_dir rawDepth --resultspth /p/blurdepth/models/defnet/ --geometry_model vpd --resume_geometry_from /p/blurdepth/models/vpd_depth_480x480.pth
