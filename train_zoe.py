import torch
import test
from configs.train_options import TrainOptions
from dataset.base_dataset import get_dataset
from tqdm import tqdm
import utils
from pprint import pprint
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
from models.zoedepth.trainers.builder import get_trainer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"


def fix_random_seed(seed: int):
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main_worker(gpu, ngpus_per_node, conf):
    try:
        seed = conf.seed if 'seed' in conf and conf.seed else 43
        fix_random_seed(seed)

        conf.gpu = gpu

        from models.zoedepth.models.builder import build_model
        model = build_model(conf,'train')
        model=model.to(DEVICE)

        model = utils.parallelize(conf, model)

        total_params = f"{round(utils.count_parameters(model)/1e6,2)}M"
        conf.total_params = total_params
        print(f"Total parameters : {total_params}")

        
        dataset_kwargs = {'dataset_name': conf.dataset, 'data_path': conf.data_path,'rgb_dir':conf.rgb_dir, 'depth_dir':conf.depth_dir}
        dataset_kwargs['crop_size'] = (conf.crop_h, conf.crop_w)

        train_dataset = get_dataset(**dataset_kwargs,is_train=True)
        val_dataset = get_dataset(**dataset_kwargs, is_train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size,
                                                num_workers=0,pin_memory=True)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                num_workers=0,pin_memory=True)

        trainer = get_trainer(conf)(
            conf, model, train_loader, val_loader, device=conf.gpu)

        trainer.train()
    finally:
        import wandb
        wandb.finish()


# Dataset setting
# dataset_kwargs = {'dataset_name': conf.dataset, 'data_path': conf.data_path,'rgb_dir':conf.rgb_dir, 'depth_dir':conf.depth_dir}
# dataset_kwargs['crop_size'] = (conf.crop_h, conf.crop_w)

# train_dataset = get_dataset(**dataset_kwargs,is_train=True)
# val_dataset = get_dataset(**dataset_kwargs, is_train=False)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size,
#                                            num_workers=0,pin_memory=True)

# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
#                                          num_workers=0,pin_memory=True)


import os
import numpy as np
import torch.multiprocessing as mp

device_id=0
opt = TrainOptions()
conf=opt.get_arg_dict()


try:
    node_str = os.environ['SLURM_JOB_NODELIST'].replace(
        '[', '').replace(']', '')
    nodes = node_str.split(',')

    conf.world_size = len(nodes)
    conf.rank = int(os.environ['SLURM_PROCID'])
    # config.save_dir = "/ibex/scratch/bhatsf/videodepth/checkpoints"

except KeyError as e:
    # We are NOT using SLURM
    conf.world_size = 1
    conf.rank = 0
    nodes = ["127.0.0.1"]


if conf.distributed:
    print(conf.rank)
    port = np.random.randint(15000, 15025)
    conf.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
    print(conf.dist_url)
    conf.dist_backend = 'nccl'
    conf.gpu = None

ngpus_per_node = torch.cuda.device_count()
conf.num_workers = conf.workers
conf.ngpus_per_node = ngpus_per_node
pprint(conf)


if conf.distributed:
        conf.world_size = ngpus_per_node * conf.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, conf))
else:
    if ngpus_per_node == 1:
        conf.gpu = 0
    main_worker(conf.gpu, ngpus_per_node, conf)



# seed = conf.seed if 'seed' in conf and conf.seed else 43
# fix_random_seed(seed)

# conf.gpu = 0
# model = utils.parallelize(conf, model)

# total_params = f"{round(utils.count_parameters(model)/1e6,2)}M"
# conf.total_params = total_params
# print(f"Total parameters : {total_params}")


# dataset_kwargs = {'dataset_name': conf.dataset, 'data_path': conf.data_path,'rgb_dir':conf.rgb_dir, 'depth_dir':conf.depth_dir}
# dataset_kwargs['crop_size'] = (conf.crop_h, conf.crop_w)

# train_dataset = get_dataset(**dataset_kwargs,is_train=True)
# val_dataset = get_dataset(**dataset_kwargs, is_train=False)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size,
#                                         num_workers=0,pin_memory=True)

# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
#                                         num_workers=0,pin_memory=True)

# trainer = get_trainer(conf)(
#     conf, model, train_loader, val_loader, device=conf.gpu)












