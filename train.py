import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim as optim
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

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


opt = TrainOptions()
args = opt.initialize().parse_args()
args.shift_window_test=True
args.flip_test=True
print(args)

#get model and load weights
device = torch.device(0)
model = VPDDepth(args=args)
cudnn.benchmark = True
model.to(device)
#load weights to the model
from collections import OrderedDict
if args.resume_from:
    model_weight = torch.load(args.resume_from)['model']
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight, strict=False)
model_params = model.parameters()

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


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3,pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,pin_memory=True)

results_dict=test.validate_single(val_loader, model, device, args,lowGPU=True)
print(results_dict)

# criterion_d = SiLogLoss()
# optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
#                 constructor='LDMOptimizerConstructor',
#                 paramwise_cfg=dict(layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))

optimizer = optim.AdamW(model_params,lr=0.000001, betas=(0.9, 0.999))
criterion_d = SiLogLoss()
criterion_b=MSELoss()
model.train()
#iterate though dataset
for i in range(1000):
    total_d_loss,total_b_loss=0,0
    for batch_idx, batch in enumerate(train_loader):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        class_id = batch['class_id']
        gt_blur = batch['blur'].to(device)

        preds = model(input_RGB, class_ids=class_id)
        optimizer.zero_grad()
        loss_d=criterion_d(preds['pred_d'].squeeze(dim=1), depth_gt)
        loss_b=criterion_b(preds['blur'],gt_blur)
        loss=loss_d+loss_b
        total_d_loss+=loss_d.item()
        total_b_loss+=loss_b.item()
        loss.backward()
        optimizer.step()
        if batch_idx%50==0:
             print("batch idx=%2d" %(batch_idx))
    print("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" %(i,total_b_loss/len(train_loader),total_d_loss/len(train_loader)))      
    if i%10==0:
        results_dict=test.validate_single(val_loader, model, device, args,lowGPU=True)
        print(results_dict)