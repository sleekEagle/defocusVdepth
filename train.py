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
import utils_depth.logging as logging

from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions
import utils

from tensorboardX import SummaryWriter

# from models_depth.optimizer import build_optimizers
from utils_depth.criterion import SiLogLoss,MSELoss
from configs.train_options import TrainOptions

import test
import importlib
importlib.reload(test)
import time

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


opt = TrainOptions()
args = opt.initialize().parse_args()
args.shift_window_test=True
args.flip_test=True
print(args)

#init distributed training
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size=dist.get_world_size()
print(f"Start running training on rank {rank}.")
device_id = rank % torch.cuda.device_count()

model = VPDDepth(args=args).to(rank)
#get model and load weights
print('loading weigths to the model....')
cudnn.benchmark = True
#load weights to the model
from collections import OrderedDict
model_weight = torch.load(args.resume_from)['model']
if 'module' in next(iter(model_weight.items()))[0]:
    model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
model.load_state_dict(model_weight, strict=False)
print('loaded weights')
#make the model distributed
model = DDP(model, device_ids=[device_id])
model_params = model.parameters()
dist.barrier()
print('barrier 1 released.')
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
dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path,'rgb_dir':args.rgb_dir, 'depth_dir':args.depth_dir,'fdist':3.0}
dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

train_dataset = get_dataset(**dataset_kwargs,is_train=True)
val_dataset = get_dataset(**dataset_kwargs, is_train=False)

sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, 
    )
sampler_val = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=1, rank=0, shuffle=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,sampler=sampler_train,
                                           num_workers=1,pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         num_workers=0,pin_memory=True)


# criterion_d = SiLogLoss()
# optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
#                 constructor='LDMOptimizerConstructor',
#                 paramwise_cfg=dict(layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))

criterion_d = SiLogLoss()
criterion_b=MSELoss()
print('validating...')
results_dict,loss_d=test.validate(val_loader, model, criterion_d, device_id, args)
print(results_dict)

dist.barrier()
print('**************passed barrier**************** rank='+str(rank))
print('lr='+str(args.max_lr))
print('lr type='+str(type(args.max_lr)))
optimizer = optim.AdamW(model_params,lr=args.max_lr, betas=(0.9, 0.999))
model.train()
#iterate though dataset
print('train_loader len='+str(len(train_loader)))
for i in range(1000):
    total_d_loss,total_b_loss=0,0
    start = time.time()
    for batch_idx, batch in enumerate(train_loader):
        input_RGB = batch['image'].to(device_id)
        depth_gt = batch['depth'].to(device_id)
        class_id = batch['class_id']
        gt_blur = batch['blur'].to(device_id)

        preds = model(input_RGB, class_ids=class_id)
        optimizer.zero_grad()
        loss_d=criterion_d(preds['pred_d'].squeeze(dim=1), depth_gt)
        loss_b=criterion_b(preds['blur'],gt_blur)
        loss_b*=0.1
        loss=loss_d+loss_b
        total_d_loss+=loss_d.item()
        total_b_loss+=loss_b.item()
        loss.backward()
        optimizer.step()
        #print("batch idx=%2d" %(batch_idx))
    print("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" %(i,total_b_loss/len(train_loader),total_d_loss/len(train_loader)))  
    end = time.time()
    #print("Elapsed time = %11.1f" %(end-start))    
    if i%10==0:
        print('validating...')
        results_dict,loss_d=test.validate(val_loader, model, criterion_d, device_id, args)
        print(results_dict)
