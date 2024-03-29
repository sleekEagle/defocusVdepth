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
logger=logging
from os.path import join

from models_depth.AENET import AENet
from models_depth.midas import MidasCore
from models_depth.model import VPDDepth
from models_depth.Selector import Selector


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
dataset=args.rgb_dir[10:]
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S_")+dataset+'_comb_train.log'
logpath=join(args.resultspth,dt_string)

logger.basicConfig(filename=logpath,filemode='w', level=logger.INFO)
logger.info('Starting training')
logger.info(args)

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
************
load blur model
************
'''
if args.blur_model=='defnet':
    ch_inp_num = 3
    ch_out_num = 1
    blur_model = AENet(ch_inp_num, 1, 16, flag_step2=True).to(device_id)
elif args.blur_model=='midas':
    midasouts={}
    # midas_model_type='DPT_BEiT_L_384'
    midas_model_type='MiDaS_small'
    # midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
    use_pretrained_midas=False
    train_midas=True
    freeze_midas_bn=False
    blur_model = torch.hub.load("intel-isl/MiDaS", midas_model_type,pretrained=use_pretrained_midas).to(device_id)
    blur_model.scratch.refinenet1.register_forward_hook(get_activation("r1",midasouts))

if args.resume_blur_from:
   # loading weights of the first step
    print('loading blur model from path :'+args.resume_blur_from)
    logging.info("loading blur model from path : "+str(args.resume_blur_from))
    pretrained_dict = torch.load(args.resume_blur_from)
    blur_model.load_state_dict(pretrained_dict['state_dict'])

'''
************
load geometry model
************
'''
if args.geometry_model=='vpd':
    geometry_model = VPDDepth(args=args).to(device_id)
    #get model and load weights
    if args.resume_geometry_from:
        from collections import OrderedDict
        print('loading weigths to the model....')
        logging.info('loading weigths to the model....')
        cudnn.benchmark = True
        #load weights to the model
        print('loading from :'+str(args.resume_geometry_from))
        logging.info('loading from :'+str(args.resume_geometry_from))
        model_weight = torch.load(args.resume_geometry_from)['model']
        if 'module' in next(iter(model_weight.items()))[0]:
            model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
        geometry_model.load_state_dict(model_weight, strict=False)
        print('loaded weights')
        logging.info('loaded weights')

criterion=torch.nn.MSELoss()

selector=Selector(blur_model=blur_model,geometry_model=geometry_model)

'''
Evauate the models
'''
# logger.info('validating the blur model...')
# results_dict=test.validate_dist(val_loader, blur_model, criterion, device_id, args,min_dist=0.0,max_dist=2.0,model_name=args.blur_model)
# print("blur model error dist : 0-2 " + str(results_dict))
# logger.info("blur model error dist : 0-2 " + str(results_dict))

# logger.info('validating the geometric model...')
# results_dict=test.validate_dist(val_loader, geometry_model, criterion, device_id, args,min_dist=2.0,max_dist=10.0,model_name=args.geometry_model)
# print("geo model error dist : 2-10 " + str(results_dict))
# logger.info("geo model error dist : 2-10 " + str(results_dict))



evalitr=10

# logger.info('validating the selectorNet model...')
# results_dict=test.validate_dist(val_loader, selectorNet, criterion, device_id, args,min_dist=0.0,max_dist=2.0,model_name="combined")
# print("dist : 0-2 " + str(results_dict))
# logger.info("dist : 0-2 " + str(results_dict))

# results_dict=test.validate_dist(val_loader, selectorNet, criterion, device_id, args,min_dist=2.0,max_dist=10.0,model_name="combined")
# print("dist : 2-10 " + str(results_dict))
# logger.info("dist : 2-10 " + str(results_dict))

#evaluate the selector model
results_dict=test.validate_dist(val_loader, selector, criterion, device_id, args,min_dist=0.0,max_dist=2.0,model_name="combined",lowGPU=True)

for i in range(600):
    total_d_loss=0
    for batch_idx, batch in enumerate(train_loader):
        input_RGB = batch['image'].to(device_id)
        depth_gt = batch['depth'].to(device_id)
        class_id = batch['class_id']
        gt_blur = batch['blur'].to(device_id)

        selector(input_RGB,class_id)

        with torch.no_grad():
            #predict depth with the geometry model
            pred_geo = geometry_model(input_RGB, class_ids=class_id)
            pred_geo=pred_geo['pred_d']
            #predict with the blur model
            pred_blur,_=blur_model(input_RGB)

        to_blur_model=pred_geo<2.0
        pred_geo[to_blur_model]=0.
        pred_blur[~to_blur_model]=0.
        comb_pred=pred_geo+pred_blur

        mask=(depth_gt>0.0).detach_()
        gt_selection=torch.zeros_like(selector_pred)
        gt_selection[depth_gt>2.0]=1.0
        # weight_mask=(depth_gt<2.0).int()
        # gt_weight=torch.zeros_like(weight_pred)
        # gt_weight[:,0,:,:]=weight_mask
        loss=criterion(gt_selection,selector_pred)
        # loss=criterion(depth_pred.squeeze(dim=1)[mask], depth_gt[mask])
        total_d_loss+=loss.item()
        loss.backward()
        optimizer.step()
    print("Epochs=%3d depth loss=%5.4f" %(i,total_d_loss/len(train_loader)))  
    logging.info("Epochs=%3d depth loss=%5.4f",i,total_d_loss/len(train_loader))
    # if (i+1)%evalitr==0:
    #     logger.info('validating the selectorNet model...')
    #     results_dict=test.validate_dist(val_loader, selectorNet, criterion, device_id, args,min_dist=0.0,max_dist=2.0,model_name="combined")
    #     print("dist : 0-2 " + str(results_dict))
    #     logger.info("dist : 0-2 " + str(results_dict))

    #     results_dict=test.validate_dist(val_loader, selectorNet, criterion, device_id, args,min_dist=2.0,max_dist=10.0,model_name="combined")
    #     print("dist : 2-10 " + str(results_dict))
    #     logger.info("dist : 2-10 " + str(results_dict))

    #     #save model
    #     torch.save({'state_dict': selectorNet.conv_selector.state_dict()},
    #                 os.path.join(os.path.abspath(args.resultspth),('selector_'+args.rgb_dir)+'.tar'))

