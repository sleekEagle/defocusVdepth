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
sys.path.append('stable-diffusion')
from models_depth.model import VPDDepth
import utils_depth.metrics as metrics
import logging

from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions
import utils

from tensorboardX import SummaryWriter

# from models_depth.optimizer import build_optimizers
from utils_depth.criterion import SiLogLoss,MSELoss
from configs.train_options import TrainOptions

import test
import importlib
import time
from os.path import join

from models_depth.AENET import AENet



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
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")+'_VPD_eval.log'
logpath=join(args.resultspth,dt_string)
logging.basicConfig(filename=logpath,filemode='w', level=logging.INFO)
logging.info('Starting training')
logging.info(args)

model = VPDDepth(args=args).to(device_id)
model_params = model.parameters()

#get model and load weights
if args.resume_from:
    from collections import OrderedDict
    print('loading weigths to the model....')
    logging.info('loading weigths to the model....')
    cudnn.benchmark = True
    #load weights to the model
    print('loading from :'+str(args.resume_from))
    logging.info('loading from :'+str(args.resume_from))
    model_weight = torch.load(args.resume_from)['model']
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight, strict=False)
    print('loaded weights')
    logging.info('loaded weights')

#create the name of the model
pretrain = args.pretrained.split('.')[0]
maxlrstr = str(args.max_lr).replace('.', '')
minlrstr = str(args.min_lr).replace('.', '')
layer_decaystr = str(args.layer_decay).replace('.', '')
weight_decaystr = str(args.weight_decay).replace('.', '')
num_filter = str(args.num_filters[0]) if args.num_deconv > 0 else ''
num_kernel = str(args.deconv_kernels[0]) if args.num_deconv > 0 else ''
name = [args.dataset, str(args.batch_size), pretrain.split('/')[-1], 'deconv'+str(args.num_deconv), \
        str(num_filter), str(num_kernel), str(args.crop_h), str(args.crop_w), maxlrstr, minlrstr, \
        layer_decaystr, weight_decaystr, str(args.epochs)]
if args.exp_name != '':
        name.append(args.exp_name)
exp_name = '_'.join(name)
print('This experiments: ', exp_name)

# Dataset setting
dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path,'rgb_dir':args.rgb_dir, 'depth_dir':args.depth_dir,'is_blur':args.is_blur}
dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

val_dataset = get_dataset(**dataset_kwargs, is_train=False)

sampler_val = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=1, rank=0, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         num_workers=0,pin_memory=True)


criterion_d = SiLogLoss()
print('validating...')
def vali_dist():
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=0.0,max_dist=1.0,model_name=args.geometry_model)
        print("dist : 0-1 " + str(results_dict))
        logging.info("dist : 0-1 " + str(results_dict))

        results_dict,loss_d=test.validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=1.0,max_dist=2.0,model_name=args.geometry_model)
        print("dist : 1-2 " + str(results_dict))
        logging.info("dist : 1-2 " + str(results_dict))

        results_dict,loss_d=test.validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=2.0,max_dist=3.0,model_name=args.geometry_model)
        print("dist : 2-3 " + str(results_dict))
        logging.info("dist : 2-3 " + str(results_dict))

        results_dict,loss_d=test.validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=3.0,max_dist=4.0,model_name=args.geometry_model)
        print("dist : 3-4 " + str(results_dict))
        logging.info("dist : 3-4 " + str(results_dict))

        results_dict,loss_d=test.validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=4.0,max_dist=5.0,model_name=args.geometry_model)
        print("dist : 4-5 " + str(results_dict))
        logging.info("dist : 4-5 " + str(results_dict))

        results_dict,loss_d=test.validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=5.0,max_dist=6.0,model_name=args.geometry_model)
        print("dist : 5-6 " + str(results_dict))
        logging.info("dist : 5-6 " + str(results_dict))

        results_dict,loss_d=test.validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=6.0,max_dist=8.0,model_name=args.geometry_model)
        print("dist : 6-8 " + str(results_dict))
        logging.info("dist : 6-8 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=8.0,max_dist=10.0,model_name=args.geometry_model)
        print("dist : 8-10 " + str(results_dict))
        logging.info("dist : 8-10 " + str(results_dict))


if __name__ == "__main__":
    vali_dist()
