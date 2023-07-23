import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
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
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S_")+args.blur_model+'.log'
logpath=join(args.resultspth,dt_string)

# logger= logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# handler = logging.FileHandler(logpath, 'a', 'utf-8')
# handler.setFormatter(logging.Formatter(": %(levelname)s:%(asctime)s | %(message)s",datefmt='%m/%d/%Y %I:%M:%S %p'))
# logger.addHandler(handler)


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
import AENET used for defocus blur based depth prediction
'''
def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

if args.blur_model=='defnet':
    ch_inp_num = 3
    ch_out_num = 1
    model = AENet(ch_inp_num, 1, 16, flag_step2=True).to(device_id)
elif args.blur_model=='midas':
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

model_params = model.parameters()
criterion=torch.nn.MSELoss()
#load the saved weights to the model
print('resume from :'+str(args.resume_blur_from))

args.resume_blur_from='C:\\Users\\lahir\\Documents\\refocused_f_25_fdist_2.tar'
if args.resume_blur_from:
    # loading weights of the first step
    print('loading model....')
    logging.info("loading model....")
    print('model path :'+args.resume_blur_from)
    logging.info("model path : "+str(args.resume_blur_from))
    pretrained_dict = torch.load(args.resume_blur_from)
    model.load_state_dict(pretrained_dict['state_dict'])
    model.eval()
    #evaluating the loaded model
    print('evaluating model...')
    logger.info('evaluating model...')
    results_dict1,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=0.0,max_dist=1.0,model_name=args.blur_model)
    print("dist : 0-1 " + str(results_dict1))
    logger.info("dist : 0-1 " + str(results_dict1))
    results_dict2,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=1.0,max_dist=2.0,model_name=args.blur_model)
    print("dist : 1-2 " + str(results_dict2))
    logger.info("dist : 1-2 " + str(results_dict2))

print('lr='+str(args.max_lr))
optimizer = optim.Adam(model_params,lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
model.train()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#iterate though dataset
print('train_loader len='+str(len(train_loader)))
logging.info('train_loader len=%s',str(len(train_loader)))
evalitr=10
best_loss=0
for i in range(600):
    total_d_loss,total_b_loss=0,0
    for batch_idx, batch in enumerate(train_loader):
        input_RGB = batch['image'].to(device_id)
        depth_gt = batch['depth'].to(device_id)
        class_id = batch['class_id']
        gt_blur = batch['blur'].to(device_id)

        if args.blur_model=='defnet':
            depth_pred,blur_pred = model(input_RGB,flag_step2=True)
        elif args.blur_model=='midas':
            # blur_pred,depth_pred,_=model(input_RGB,return_rel_depth=True)
            depth_pred=model(input_RGB)
            blur_pred=midasouts['r1'][:,0,:,:]
            blur_pred=torch.unsqueeze(blur_pred,dim=1)
            blur_pred=torch.nn.functional.interpolate(blur_pred,size=(input_RGB.shape[-2],input_RGB.shape[-2]),mode='bilinear')
            blur_pred=torch.squeeze(blur_pred,dim=1)

        optimizer.zero_grad()

        mask=(depth_gt>0.0)*(depth_gt<2.0).detach_()
        loss_d=criterion(depth_pred.squeeze(dim=1)[mask], depth_gt[mask])
        loss_b=criterion(blur_pred.squeeze(dim=1)[mask],gt_blur[mask])
        if(torch.isnan(loss_d) or torch.isnan(loss_b)):
            # print('nan in losses')
            logging.info('nan in losses')
            continue
        loss=loss_d+loss_b
        total_d_loss+=loss_d.item()
        total_b_loss+=loss_b.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" %(i,total_b_loss/len(train_loader),total_d_loss/len(train_loader)))  
    logging.info("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" , i,total_b_loss/len(train_loader),total_d_loss/len(train_loader))

    #print("Elapsed time = %11.1f" %(end-start))    
    if (i+1)%evalitr==0:
        # print('validating...')
        # results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=0.0,max_dist=1.0,model_name="def")
        # print(results_dict)
        # rmse=vali_dist()
        model.eval()
        rmse_total=0
        n=0
        with torch.no_grad():
            # for batch_idx, batch in enumerate(val_loader):
            #     input_RGB = batch['image'].to(device_id)
            #     depth_gt = batch['depth'].to(device_id)
            #     class_id = batch['class_id']
            #     gt_blur = batch['blur'].to(device_id)

            #     s1_fcs = torch.ones([input_RGB.shape[0],1, input_RGB.shape[2], input_RGB.shape[3]])
            #     s1_fcs*=args.fdist
            #     s1_fcs = s1_fcs.float().to(device_id)
            #     depth_pred,blur_pred = def_model(input_RGB,flag_step2=True,x2=s1_fcs)

            #     mask=(torch.squeeze(depth_gt)>0)*(torch.squeeze(depth_gt)<2).detach_()
            #     #calc rmse
            #     diff=torch.squeeze(depth_gt)-torch.psqueeze(depth_pred)
            #     rmse=torch.sqrt(torch.mean(torch.pow(diff[mask],2))).item()
            #     if(rmse!=rmse):
            #         continue
            #     rmse_total+=rmse
            #     n+=1
            # print("val RMSE = %2.5f" %(rmse_total/n))
            # logging.info("val RMSE = " +str(rmse_total/n))
            # results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=0.0,max_dist=1.0,model_name="def")
            results_dict1,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=0.0,max_dist=1.0,model_name=args.blur_model)
            print("dist : 0-1 " + str(results_dict1))
            logger.info("dist : 0-1 " + str(results_dict1))
            results_dict2,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=1.0,max_dist=2.0,model_name=args.blur_model)
            print("dist : 1-2 " + str(results_dict2))
            logger.info("dist : 1-2 " + str(results_dict2))
            # vali_dist()

            rmse=(results_dict1['rmse']+results_dict2['rmse'])*0.5
            if(i+1==evalitr):
                best_loss=rmse
            else:
                if rmse<best_loss:
                    best_loss=rmse
                    #save model
                    torch.save({
                    'epoch': i + 1,
                    'iters': i + 1,
                    'best': best_loss,
                    'state_dict': model.state_dict(),
                    'optimize':optimizer.state_dict(),
                    },  os.path.join(os.path.abspath(args.resultspth),(args.blur_model+'_'+args.rgb_dir)+'.tar'))
                    logging.info("saved model")
                    print('model saved')
            #get learning rate
            current_lr=get_lr(optimizer)
            print('current learning rate='+str(current_lr))
            logging.info("current learning rate="+str(current_lr))
        model.train()

            

