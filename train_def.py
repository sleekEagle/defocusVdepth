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
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")+'.log'
logpath=join(args.resultspth,dt_string)

logging.basicConfig(filename=logpath,filemode='w', level=logging.INFO)
logging.info('Starting training')
logging.info(args)

# Dataset setting
dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path,'rgb_dir':args.rgb_dir, 'depth_dir':args.depth_dir}
dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

train_dataset = get_dataset(**dataset_kwargs,is_train=True)
val_dataset = get_dataset(**dataset_kwargs, is_train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           num_workers=0,pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         num_workers=0,pin_memory=True)

'''
Load models
'''
criterion=torch.nn.MSELoss()
print('lr='+str(args.max_lr))
assert args.blur_model in ['defnet','midas'],'blur model should be either defnet,midas'
#load model
if args.blur_model == 'defnet':
    from models_depth.AENET import AENet
    ch_inp_num = 3
    ch_out_num = 1
    model = AENet(ch_inp_num, 1, 16, flag_step2=True).to(device_id)    
elif args.blur_model=='midas':
    from models_depth.midas import Midas
    model=Midas(layers=['l4_rn'],model_type=args.midas_type)
    model_params = model.parameters()

model.to(device_id)
model_params = model.parameters()
optimizer = optim.Adam(model_params,lr=0.0001)
model.train()

#iterate though dataset
print('train_loader len='+str(len(train_loader)))
logging.info('train_loader len=%s',str(len(train_loader)))
evalitr=10
best_loss=0
for i in range(1000):
    total_d_loss,total_b_loss=0,0
    start = time.time()
    for batch_idx, batch in enumerate(train_loader):
        input_RGB = batch['image'].to(device_id)
        depth_gt = batch['depth'].to(device_id)
        class_id = batch['class_id']
        gt_blur = batch['blur'].to(device_id)

        depth_pred,blur_pred = model(input_RGB)

        optimizer.zero_grad()

        mask=(depth_gt>0)*(depth_gt<2).detach_()
        loss_d=criterion(depth_pred.squeeze(dim=1)[mask], depth_gt[mask])
        loss_b=criterion(blur_pred.squeeze(dim=1)[mask],gt_blur[mask])
        if(torch.isnan(loss_d) or torch.isnan(loss_b)):
            continue
        loss=loss_d+loss_b
        total_d_loss+=loss_d.item()
        total_b_loss+=loss_b.item()
        loss.backward()
        optimizer.step()
    print("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" %(i,total_b_loss/len(train_loader),total_d_loss/len(train_loader)))  
    logging.info("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" , i,total_b_loss/len(train_loader),total_d_loss/len(train_loader))
    end = time.time()    

    #print("Elapsed time = %11.1f" %(end-start))    
    if (i+1)%evalitr==0:
        model.eval()
        rmse_total=0
        n=0
        with torch.no_grad():
            results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=0.0,max_dist=2.0,model_name="def")
            print("dist : 0-2 " + str(results_dict))
            logging.info("dist : 0-2 " + str(results_dict))
            torch.save({
                    'state_dict': model.state_dict(),
                    },  os.path.join(os.path.abspath(args.resultspth),args.rgb_dir[10:]+'_'+args.blur_model+'.tar'))
            logging.info("saved model")
            print('model saved')
        model.train()

            

