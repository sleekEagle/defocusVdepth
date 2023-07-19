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
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")+'.log'
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
if args.model_name=='defnet':
    ch_inp_num = 3
    ch_out_num = 1
    model = AENet(ch_inp_num, 1, 16, flag_step2=True).to(device_id)
elif args.model_name=='midas':
    midas_model_type='DPT_BEiT_L_384'
    use_pretrained_midas=False
    train_midas=True
    freeze_midas_bn=False
    model = MidasCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
                                train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn,
                                img_size_in=480,img_size_out=480).to(device_id)

model_params = model.parameters()
criterion=torch.nn.MSELoss()
#load the saved weights to the model
print('resume from :'+str(args.resume_from))
if args.resume_from:
    # loading weights of the first step
    print('loading model....')
    logging.info("loading model....")
    print('model path :'+args.resume_from)
    logging.info("model path : "+str(args.resume_from))
    pretrained_dict = torch.load(args.resume_from)
    model_dict = model.state_dict()
    for param_tensor in model_dict:
        for param_pre in pretrained_dict:
            if param_tensor == param_pre:
                model_dict.update({param_tensor: pretrained_dict[param_pre]})
    model.load_state_dict(model_dict)

    #evaluating the loaded model
    print('evaluating model...')
    logging.info('evaluating model...')
    results_dict1,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=0.0,max_dist=1.0,model_name="def")
    print("dist : 0-1 " + str(results_dict1))
    logging.info("dist : 0-1 " + str(results_dict1))
    results_dict2,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=1.0,max_dist=2.0,model_name="def")
    print("dist : 1-2 " + str(results_dict2))
    logging.info("dist : 1-2 " + str(results_dict2))


# print('validating...')
def vali_dist():
    if device_id==0:
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=0.0,max_dist=1.0,model_name="def")
        print("dist : 0-1 " + str(results_dict))
        logging.info("dist : 0-1 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=1.0,max_dist=2.0,model_name="def")
        print("dist : 1-2 " + str(results_dict))
        logging.info("dist : 1-2 " + str(results_dict))

        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=2.0,max_dist=3.0,model_name="def")
        print("dist : 2-3 " + str(results_dict))
        logging.info("dist : 2-3 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=3.0,max_dist=4.0,model_name="def")
        print("dist : 3-4 " + str(results_dict))
        logging.info("dist : 3-4 " + str(results_dict))
        
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=4.0,max_dist=5.0,model_name="def")
        print("dist : 4-5 " + str(results_dict))
        logging.info("dist : 4-5 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=5.0,max_dist=6.0,model_name="def")
        print("dist : 5-6 " + str(results_dict))
        logging.info("dist : 5-6 " + str(results_dict))
        
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=6.0,max_dist=8.0,model_name="def")
        print("dist : 6-8 " + str(results_dict))
        logging.info("dist : 6-8 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=8.0,max_dist=10.0,model_name="def")
        print("dist : 8-10 " + str(results_dict))
        logging.info("dist : 8-10 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=0.0,max_dist=10.0,model_name="def")
        print("dist : 0-10 " + str(results_dict))
        logging.info("dist : 0-10 " + str(results_dict))
        return results_dict['rmse']
    if device_id==1:
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=2.0,max_dist=3.0,model_name="def")
        print("dist : 2-3 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=3.0,max_dist=4.0,model_name="def")
        print("dist : 3-4 " + str(results_dict))

    if device_id==2:
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=4.0,max_dist=5.0,model_name="def")
        print("dist : 4-5 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=5.0,max_dist=6.0,model_name="def")
        print("dist : 5-6 " + str(results_dict))

    if device_id==3:
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=6.0,max_dist=8.0,model_name="def")
        print("dist : 6-8 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=8.0,max_dist=10.0,model_name="def")
        print("dist : 8-10 " + str(results_dict))

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
    start = time.time()
    for batch_idx, batch in enumerate(train_loader):
        input_RGB = batch['image'].to(device_id)
        depth_gt = batch['depth'].to(device_id)
        class_id = batch['class_id']
        gt_blur = batch['blur'].to(device_id)

        if args.model_name=='defnet':
            depth_pred,blur_pred = model(input_RGB,flag_step2=True)
        elif args.model_name=='midas':
            blur_pred,depth_pred,_=model(input_RGB,return_rel_depth=True)

        optimizer.zero_grad()

        mask=(depth_gt>0.0)*(depth_gt<2.0).detach_()
        loss_d=criterion(depth_pred.squeeze(dim=1)[mask], depth_gt[mask])
        loss_b=criterion(blur_pred.squeeze(dim=1)[mask],gt_blur[mask])
        if(torch.isnan(loss_d) or torch.isnan(loss_b)):
            print('nan in losses')
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
    end = time.time()    

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
            results_dict1,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=0.0,max_dist=1.0,model_name="def")
            print("dist : 0-1 " + str(results_dict1))
            logging.info("dist : 0-1 " + str(results_dict1))
            results_dict2,loss_d=test.validate_dist(val_loader, model, criterion, device_id, args,min_dist=1.0,max_dist=2.0,model_name="def")
            print("dist : 1-2 " + str(results_dict2))
            logging.info("dist : 1-2 " + str(results_dict2))
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
                    },  os.path.join(os.path.abspath(args.resultspth),(args.model_name+'_'+args.rgb_dir)+'.tar'))
                    logging.info("saved model")
                    print('model saved')
            #get learning rate
            current_lr=get_lr(optimizer)
            print('current learning rate='+str(current_lr))
            logging.info("current learning rate="+str(current_lr))
        model.train()

            

