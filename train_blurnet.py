import torch
from dataset.base_dataset import get_dataset
from tqdm import tqdm
import utils
from pprint import pprint
from models.trainers.builder import get_trainer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import hydra
from omegaconf import DictConfig, OmegaConf,open_dict
import os
import numpy as np
import torch.multiprocessing as mp
from models.blurnet.BlurNet import BlurNet

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"


from models.blurnet import BlurNet


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

        model=BlurNet.build()
        model=model.to(DEVICE)


        model = utils.parallelize(conf, model)

        total_params = f"{round(utils.count_parameters(model)/1e6,2)}M"
        conf.total_params = total_params
        print(f"Total parameters : {total_params}")

        
        dataset_kwargs = {'dataset_name': conf.datasets.nyudepthv2.dataset, 'data_path': conf.datasets.nyudepthv2.data_path,'rgb_dir':conf.datasets.nyudepthv2.rgb_dir, 'depth_dir':conf.datasets.nyudepthv2.depth_dir}
        dataset_kwargs['crop_size'] = (conf.models[conf.common.train.image_model].train.input_height, conf.models[conf.common.train.image_model].train.input_width)

        train_dataset = get_dataset(**dataset_kwargs,is_train=True)
        val_dataset = get_dataset(**dataset_kwargs, is_train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.common.train.batch_size,
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

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_train(conf : DictConfig):
    OmegaConf.set_struct(conf, True)
    with open_dict(conf):
        try:
            node_str = os.environ['SLURM_JOB_NODELIST'].replace(
                '[', '').replace(']', '')
            nodes = node_str.split(',')

            conf.common.world_size = len(nodes)
            conf.common.rank = int(os.environ['SLURM_PROCID'])
            # config.save_dir = "/ibex/scratch/bhatsf/videodepth/checkpoints"

        except KeyError as e:
            # We are NOT using SLURM
            conf.common.world_size = 1
            conf.common.rank = 0
            nodes = ["127.0.0.1"]

        if conf.common.train.distributed:
            port = np.random.randint(15000, 15025)
            conf.common.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
            conf.common.dist_backend = 'nccl'
            conf.common.train.gpu = None

        ngpus_per_node = torch.cuda.device_count()
        conf.common.ngpus_per_node = ngpus_per_node
        pprint(conf)

        if conf.common.train.distributed:
                conf.common.world_size = ngpus_per_node * conf.common.world_size
                mp.spawn(main_worker, nprocs=ngpus_per_node,
                        args=(ngpus_per_node, conf))
        else:
            if ngpus_per_node == 1:
                conf.common.train.gpu = 0
            OmegaConf.set_struct(conf, False)
            main_worker(conf.common.train.gpu, ngpus_per_node, conf)

if __name__ == "__main__":
    run_train()

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












