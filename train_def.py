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

from models_depth.AENET import AENet



metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

device_id=0
opt = TrainOptions()
args = opt.initialize().parse_args()
args.shift_window_test=True
args.flip_test=True
print(args)

# Dataset setting
dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path,'rgb_dir':args.rgb_dir, 'depth_dir':args.depth_dir, 'fdist':args.fdist}
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

ch_inp_num = 3
ch_out_num = 1
def_model = AENet(ch_inp_num, 1, 16, flag_step2=True).to(device_id)
model_params = def_model.parameters()

# criterion_d = SiLogLoss()
# optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
#                 constructor='LDMOptimizerConstructor',
#                 paramwise_cfg=dict(layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))

# import torch
# import torch.nn as nn

# class SiLogLoss(nn.Module):
#     def __init__(self, lambd=0.5):
#         super().__init__()
#         self.lambd = lambd

#     def forward(self, pred, target):
#         valid_mask = (target > 0).detach()
#         diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
#         loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
#                           self.lambd * torch.pow(diff_log.mean(), 2))

#         return loss
criterion_d = SiLogLoss()
criterion_b=MSELoss()

# print('validating...')
def vali_dist():
    if device_id==0:
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=0.0,max_dist=1.0,model_name="def")
        print("dist : 0-1 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=1.0,max_dist=2.0,model_name="def")
        print("dist : 1-2 " + str(results_dict))

        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=2.0,max_dist=3.0,model_name="def")
        print("dist : 2-3 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=3.0,max_dist=4.0,model_name="def")
        print("dist : 3-4 " + str(results_dict))
        
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=4.0,max_dist=5.0,model_name="def")
        print("dist : 4-5 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=5.0,max_dist=6.0,model_name="def")
        print("dist : 5-6 " + str(results_dict))
        
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=6.0,max_dist=8.0,model_name="def")
        print("dist : 6-8 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=8.0,max_dist=10.0,model_name="def")
        print("dist : 8-10 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=0.0,max_dist=10.0,model_name="def")
        print("dist : 0-10 " + str(results_dict))
    if device_id==1:
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=2.0,max_dist=3.0,model_name="def")
        print("dist : 2-3 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=3.0,max_dist=4.0,model_name="def")
        print("dist : 3-4 " + str(results_dict))

    if device_id==2:
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=4.0,max_dist=5.0,model_name="def")
        print("dist : 4-5 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=5.0,max_dist=6.0,model_name="def")
        print("dist : 5-6 " + str(results_dict))

    if device_id==3:
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=6.0,max_dist=8.0,model_name="def")
        print("dist : 6-8 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=8.0,max_dist=10.0,model_name="def")
        print("dist : 8-10 " + str(results_dict))

# print('lr='+str(args.max_lr))
# print('lr type='+str(type(args.max_lr)))
optimizer = optim.AdamW(model_params,lr=args.max_lr, betas=(0.9, 0.999))
def_model.train()



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

        depth_pred,blur_pred = def_model(input_RGB,flag_step2=True)
        optimizer.zero_grad()


        # pred=depth_pred.squeeze(dim=1)
        # target=depth_gt
        # valid_mask = (target > 0).detach()

        # diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        # loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
        #                   0.5 * torch.pow(diff_log.mean(), 2))


        loss_d=criterion_d(depth_pred.squeeze(dim=1), depth_gt)
        loss_b=criterion_b(blur_pred.squeeze(dim=1),gt_blur)
        if(torch.isnan(loss_d) or torch.isnan(loss_b)):
            print('nan in losses')
            break
        loss=loss_d+loss_b
        total_d_loss+=loss_d.item()
        total_b_loss+=loss_b.item()
        loss.backward()
        optimizer.step()
        #print("batch idx=%2d" %(batch_idx))
    print("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" %(i,total_b_loss/len(train_loader),total_d_loss/len(train_loader)))  
    end = time.time()
    #print("Elapsed time = %11.1f" %(end-start))    
    if (i+1)%10==0:
        # print('validating...')
        # results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=0.0,max_dist=1.0,model_name="def")
        # print(results_dict)
        vali_dist()
