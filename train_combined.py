import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys 
#include stable diffision path

from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions

# from models_depth.optimizer import build_optimizers
from utils_depth.criterion import SiLogLoss,MSELoss
from configs.train_options import TrainOptions

import test
import importlib
import time
import logging
from os.path import join

from models_depth.AENET import AENet
from models_depth.midas import MidasCore



metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

device_id=0
opt = TrainOptions()
args = opt.initialize().parse_args()
args.shift_window_test=True
args.flip_test=True
print(args)

#setting up logging
if not os.path.exists(args.resultspth):
    os.makedirs(args.resultspth)
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S_")+args.model_name+'.log'
logpath=join(args.resultspth,dt_string)

# logger= logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# handler = logging.FileHandler(logpath, 'a', 'utf-8')
# handler.setFormatter(logging.Formatter(": %(levelname)s:%(asctime)s | %(message)s",datefmt='%m/%d/%Y %I:%M:%S %p'))
# logger.addHandler(handler)


logging.basicConfig(filename=logpath,filemode='w', level=logging.INFO)
logging.info('Starting training')
logging.info(args)

# Dataset setting
dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path,'rgb_dir':args.rgb_dir, 'depth_dir':args.depth_dir,'is_blur':args.is_blur}
dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

train_dataset = get_dataset(**dataset_kwargs,is_train=True)
val_dataset = get_dataset(**dataset_kwargs, is_train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           num_workers=0,pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         num_workers=0,pin_memory=True)

'''
import AENET used for defocus blur based depth prediction
'''
def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

if args.model_name=='defnet':
    ch_inp_num = 3
    ch_out_num = 1
    model = AENet(ch_inp_num, 1, 16, flag_step2=True).to(device_id)
elif args.model_name=='midas':
    midasouts={}
    # midas_model_type='DPT_BEiT_L_384'
    midas_model_type='MiDaS_small'
    # midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
    use_pretrained_midas=False
    train_midas=True
    freeze_midas_bn=False
    model = torch.hub.load("intel-isl/MiDaS", midas_model_type,pretrained=use_pretrained_midas).to(device_id)
    model.scratch.refinenet1.register_forward_hook(get_activation("r1",midasouts))
   
    # model = MidasCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
    #                             train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn,
    #                             img_size_in=256,img_size_out=256).to(device_id)