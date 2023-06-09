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
dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path,'rgb_dir':args.rgb_dir, 'depth_dir':args.depth_dir,
                  'selected_dirs':args.selected_dirs}
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
criterion=torch.nn.MSELoss()

# print('validating...')
def vali_dist():
    if device_id==0:
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=0.0,max_dist=1.0,model_name="def")
        print("dist : 0-1 " + str(results_dict))
        logging.info("dist : 0-1 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=1.0,max_dist=2.0,model_name="def")
        print("dist : 1-2 " + str(results_dict))
        logging.info("dist : 1-2 " + str(results_dict))

        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=2.0,max_dist=3.0,model_name="def")
        print("dist : 2-3 " + str(results_dict))
        logging.info("dist : 2-3 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=3.0,max_dist=4.0,model_name="def")
        print("dist : 3-4 " + str(results_dict))
        logging.info("dist : 3-4 " + str(results_dict))
        
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=4.0,max_dist=5.0,model_name="def")
        print("dist : 4-5 " + str(results_dict))
        logging.info("dist : 4-5 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=5.0,max_dist=6.0,model_name="def")
        print("dist : 5-6 " + str(results_dict))
        logging.info("dist : 5-6 " + str(results_dict))
        
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=6.0,max_dist=8.0,model_name="def")
        print("dist : 6-8 " + str(results_dict))
        logging.info("dist : 6-8 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=8.0,max_dist=10.0,model_name="def")
        print("dist : 8-10 " + str(results_dict))
        logging.info("dist : 8-10 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=0.0,max_dist=10.0,model_name="def")
        print("dist : 0-10 " + str(results_dict))
        logging.info("dist : 0-10 " + str(results_dict))
        return results_dict['rmse']
    if device_id==1:
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=2.0,max_dist=3.0,model_name="def")
        print("dist : 2-3 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=3.0,max_dist=4.0,model_name="def")
        print("dist : 3-4 " + str(results_dict))

    if device_id==2:
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=4.0,max_dist=5.0,model_name="def")
        print("dist : 4-5 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=5.0,max_dist=6.0,model_name="def")
        print("dist : 5-6 " + str(results_dict))

    if device_id==3:
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=6.0,max_dist=8.0,model_name="def")
        print("dist : 6-8 " + str(results_dict))
        results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=8.0,max_dist=10.0,model_name="def")
        print("dist : 8-10 " + str(results_dict))

print('lr='+str(args.max_lr))
optimizer = optim.Adam(model_params,lr=0.0001)
def_model.train()

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
        fdist=batch['fdist']

        s1_fcs = torch.ones([input_RGB.shape[0],1, input_RGB.shape[2], input_RGB.shape[3]])
        for fd_,fd in enumerate(fdist):
            s1_fcs[fd_,:,:,:]=fd.item()
        s1_fcs = s1_fcs.float().to(device_id)
        depth_pred,blur_pred = def_model(input_RGB,flag_step2=True,x2=s1_fcs)

        optimizer.zero_grad()

        # import matplotlib.pyplot as plt
        # img=input_RGB[0,0,:,:].cpu().numpy()
        # # img=np.swapaxes(img,0,2)
        # d=depth_gt[0,:,:].cpu().numpy()
        # b=gt_blur[0,:,:].cpu().numpy()

        # # plt.figure()
        # f, axarr = plt.subplots(3,1)
        # axarr[0].imshow(img) 
        # axarr[1].imshow(d) 
        # axarr[2].imshow(b)
        # plt.show() 




        # pred=depth_pred.squeeze(dim=1)
        # target=depth_gt
        # valid_mask = (target > 0).detach()

        # diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        # loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
        #                   0.5 * torch.pow(diff_log.mean(), 2))

        mask=(depth_gt>0)*(depth_gt<2).detach_()
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
        #print("batch idx=%2d" %(batch_idx))
    print("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" %(i,total_b_loss/len(train_loader),total_d_loss/len(train_loader)))  
    logging.info("Epochs=%3d blur loss=%5.4f  depth loss=%5.4f" , i,total_b_loss/len(train_loader),total_d_loss/len(train_loader))
    end = time.time()    

    #print("Elapsed time = %11.1f" %(end-start))    
    if (i+1)%evalitr==0:
        # print('validating...')
        # results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion_d, device_id, args,min_dist=0.0,max_dist=1.0,model_name="def")
        # print(results_dict)
        # rmse=vali_dist()
        def_model.eval()
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
            results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=0.0,max_dist=2.0,model_name="def")
            # results_dict,loss_d=test.validate_dist(val_loader, def_model, criterion, device_id, args,min_dist=0.0,max_dist=1.0,model_name="def")
            print("dist : 0-2 " + str(results_dict))
            logging.info("dist : 0-2 " + str(results_dict))
            rmse=results_dict['rmse']
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
                    'state_dict': def_model.state_dict(),
                    'optimize':optimizer.state_dict(),
                    },  os.path.abspath(args.resultspth)+'/model.tar')
                    logging.info("saved model")
                    print('model saved')
        def_model.train()

            

