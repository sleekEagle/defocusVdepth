#!/bin/bash
#SBATCH --job-name="midas"
#SBATCH --error="midas.err"
#SBATCH --output="midas.out"
#SBATCH --partition="gpu"
#SBATCH -w jaguar06
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
python train_def.py --max_lr 1e-4 --batch_size 12 --rgb_dir refocused_f_25_fdist_2 --data_path /p/blurdepth/data/ --depth_dir rawDepth --resultspth /p/blurdepth/models/defnet/ --model_name midas
