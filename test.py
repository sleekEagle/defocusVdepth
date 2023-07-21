import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
sys.path.append('stable-diffusion')
from models_depth.model import VPDDepth
import utils_depth.metrics as metrics
import utils_depth.logging as logging

from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions
import utils
import math

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

def validate_single(val_loader, model, device, args,lowGPU):
    
    # if args.save_eval_pngs or args.save_visualize:
    #     result_path = os.path.join(args.result_dir, args.exp_name)
    #     # if args.rank == 0:
    #     #     logging.check_and_make_dirs(result_path)
    #     print("Saving result images in to %s" % result_path)

    # if args.rank == 0:
    #     depth_loss = logging.AverageMeter()
    model.eval()

    ddp_logger = utils.MetricLogger()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        # filename = batch['filename'][0]
        class_id = batch['class_id']
        blur_gt=batch['blur']

        with torch.no_grad():
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert w > h and bs == 1
                interval_all = w - h
                interval = interval_all // (args.shift_size-1)
                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
                class_ids = []
                for i in range(args.shift_size):
                    sliding_images.append(input_RGB[..., :, i*interval:i*interval+h])
                    sliding_masks[..., :, i*interval:i*interval+h] += 1
                    class_ids.append(class_id)
                input_RGB = torch.cat(sliding_images, dim=0)
                class_ids = torch.cat(class_ids, dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
                class_ids = torch.cat((class_ids, class_ids), dim=0)
            
            #loop insted of sending through the network for lower memory GPUs (this is slower)
            if(lowGPU):
                num=input_RGB.shape[0]
                predlist=torch.empty((0,1,input_RGB.shape[-2],input_RGB.shape[-1])).to(device)
                for i in range(num):
                    img=torch.unsqueeze(input_RGB[i,:,:,:],dim=0)
                    c=torch.unsqueeze(class_ids[i],dim=0)
                    pred = model(img, class_ids=c)
                    pred_d = pred['pred_d']
                    predlist=torch.cat((predlist,pred_d),dim=0)
            else:
                predlist = model(input_RGB, class_ids=class_ids)['pred_d']


        # pred_d = pred['pred_d']
        if args.flip_test:
            batch_s = predlist.shape[0]//2
            pred_d = (predlist[:batch_s] + torch.flip(predlist[batch_s:], [3]))/2.0
        if args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(args.shift_size):
                pred_s[..., :, i*interval:i*interval+h] += predlist[i:i+1]
            pred_d = pred_s/sliding_masks

        pred_d = pred_d.squeeze()
        depth_gt = depth_gt.squeeze()

        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)
    
        # if args.save_eval_pngs:
        #     save_path = os.path.join(result_path, filename)
        #     if save_path.split('.')[-1] == 'jpg':
        #         save_path = save_path.replace('jpg', 'png')
        #     pred_d = pred_d.squeeze()
        #     if args.dataset == 'nyudepthv2':
        #         pred_d = pred_d.cpu().numpy() * 1000.0
        #         cv2.imwrite(save_path, pred_d.astype(np.uint16),
        #                     [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #     else:
        #         pred_d = pred_d.cpu().numpy() * 256.0
        #         cv2.imwrite(save_path, pred_d.astype(np.uint16),
        #                     [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
        # if args.save_visualize:
        #     save_path = os.path.join(result_path, filename)
        #     pred_d_numpy = pred_d.squeeze().cpu().numpy()
        #     pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
        #     pred_d_numpy = pred_d_numpy.astype(np.uint8)
        #     pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
        #     cv2.imwrite(save_path, pred_d_color)

        ddp_logger.update(**computed_result)
        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    ddp_logger.synchronize_between_processes()

    for key in result_metrics.keys():
        result_metrics[key] = ddp_logger.meters[key].global_avg

    return result_metrics


def validate(val_loader, model, criterion_d, device_id, args,model_name):

    if device_id == 0:
        depth_loss = logging.AverageMeter()
    model.eval()


    ddp_logger = utils.MetricLogger()
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device_id)
        depth_gt = batch['depth'].to(device_id)
        class_id = batch['class_id']
        fdist=batch['fdist']
        #if(batch_idx>10): break
        with torch.no_grad():
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert w > h and bs == 1
                interval_all = w - h
                interval = interval_all // (args.shift_size-1)
                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
                class_ids = []
                for i in range(args.shift_size):
                    sliding_images.append(input_RGB[..., :, i*interval:i*interval+h])
                    sliding_masks[..., :, i*interval:i*interval+h] += 1
                    class_ids.append(class_id)
                input_RGB = torch.cat(sliding_images, dim=0)
                class_ids = torch.cat(class_ids, dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
                class_ids = torch.cat((class_ids, class_ids), dim=0)
            if model_name=="def":
                s1_fcs = torch.ones([input_RGB.shape[0],1, input_RGB.shape[2], input_RGB.shape[3]])
                for fd_,fd in enumerate(fdist):
                    s1_fcs[fd_,:,:,:]=fd.item()
                s1_fcs = s1_fcs.float().to(device_id)
                pred_d,pred_b = model(input_RGB,flag_step2=True,x2=s1_fcs)
            else:
                pred = model(input_RGB, class_ids=class_ids)
                pred_d = pred['pred_d']
        if args.flip_test:
            batch_s = pred_d.shape[0]//2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0
        if args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(args.shift_size):
                pred_s[..., :, i*interval:i*interval+h] += pred_d[i:i+1]
            pred_d = pred_s/sliding_masks

        pred_d = pred_d.squeeze()
        depth_gt = depth_gt.squeeze()

        loss_d = criterion_d(pred_d.squeeze(), depth_gt)

        ddp_logger.update(loss_d=loss_d.item())

        if device_id == 0:
            depth_loss.update(loss_d.item(), input_RGB.size(0))

        #cropping_img filters out valid depth values. No zero depths after this
        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)

        #if rank == 0:
        #    save_path = os.path.join(result_dir, filename)

        #    if save_path.split('.')[-1] == 'jpg':
        #        save_path = save_path.replace('jpg', 'png')

        #    if args.save_result:
        #        if args.dataset == 'kitti':
        #            pred_d_numpy = pred_d.cpu().numpy() * 256.0
        #            cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
        #                        [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #        else:
        #            pred_d_numpy = pred_d.cpu().numpy() * 1000.0
        #            cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
        #                        [cv2.IMWRITE_PNG_COMPRESSION, 0])

        #if rank == 0:
        #    loss_d = depth_loss.avg
        #    if args.pro_bar:
        #        logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        ddp_logger.update(**computed_result)
        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    #ddp_logger.synchronize_between_processes()

    for key in result_metrics.keys():
        result_metrics[key] = ddp_logger.meters[key].global_avg

    loss_d = ddp_logger.meters['loss_d'].global_avg
    return result_metrics,loss_d

#provides distance wise error
def validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=0.0,max_dist=10.0,model_name=None):

    if device_id == 0:
        depth_loss = logging.AverageMeter()
    model.eval()


    ddp_logger = utils.MetricLogger()
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device_id)
        depth_gt = batch['depth'].to(device_id)
        class_id = batch['class_id']
        #if(batch_idx>10): break
        with torch.no_grad():
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert w > h and bs == 1
                interval_all = w - h
                interval = interval_all // (args.shift_size-1)
                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
                class_ids = []
                for i in range(args.shift_size):
                    sliding_images.append(input_RGB[..., :, i*interval:i*interval+h])
                    sliding_masks[..., :, i*interval:i*interval+h] += 1
                    class_ids.append(class_id)
                input_RGB = torch.cat(sliding_images, dim=0)
                class_ids = torch.cat(class_ids, dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
                class_ids = torch.cat((class_ids, class_ids), dim=0)
            
            if model_name=='defnet':
                pred_d,pred_b = model(input_RGB,flag_step2=True)
            elif model_name=='midas':
                pred_d=model(input_RGB)
                pred_d=torch.unsqueeze(pred_d,dim=1)
            elif model_name=='vpd':
                pred = model(input_RGB, class_ids=class_ids)
                pred_d = pred['pred_d']
            else:
                return -1
        if args.flip_test:
            batch_s = pred_d.shape[0]//2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0
        if args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(args.shift_size):
                pred_s[..., :, i*interval:i*interval+h] += pred_d[i:i+1]
            pred_d = pred_s/sliding_masks

        pred_d = (pred_d.squeeze()).squeeze()
        depth_gt = depth_gt.squeeze()

        depth_gt[depth_gt<min_dist]=0.0
        depth_gt[depth_gt>max_dist]=0.0
        #if(torch.sum(depth_gt)==0.0):
        #    print('all zero!')

        loss_d = criterion_d(pred_d, depth_gt)

        ddp_logger.update(loss_d=loss_d.item())

        if device_id == 0:
            depth_loss.update(loss_d.item(), input_RGB.size(0))

        #cropping_img filters out valid depth values. No zero depths after this
        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)
        if math.isnan(computed_result['rmse']):
            continue
        #if rank == 0:
        #    save_path = os.path.join(result_dir, filename)

        #    if save_path.split('.')[-1] == 'jpg':
        #        save_path = save_path.replace('jpg', 'png')

        #    if args.save_result:
        #        if args.dataset == 'kitti':
        #            pred_d_numpy = pred_d.cpu().numpy() * 256.0
        #            cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
        #                        [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #        else:
        #            pred_d_numpy = pred_d.cpu().numpy() * 1000.0
        #            cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
        #                        [cv2.IMWRITE_PNG_COMPRESSION, 0])

        #if rank == 0:
        #    loss_d = depth_loss.avg
        #    if args.pro_bar:
        #        logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        ddp_logger.update(**computed_result)
        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    #ddp_logger.synchronize_between_processes()

    for key in result_metrics.keys():
        result_metrics[key] = ddp_logger.meters[key].global_avg

    loss_d = ddp_logger.meters['loss_d'].global_avg
    return result_metrics,loss_d

# opt = TestOptions()
# args = opt.initialize().parse_args()
# args.shift_window_test=True
# args.flip_test=True
# print(args)

# device = torch.device(0)

# model = VPDDepth(args=args)

# cudnn.benchmark = True
# model.to(device)
# from collections import OrderedDict
# model_weight = torch.load(args.ckpt_dir)['model']
# if 'module' in next(iter(model_weight.items()))[0]:
#     model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
# model.load_state_dict(model_weight, strict=False)

# model.eval()

# # img=torch.rand((1,3,480,480))
# # img=img.to(device)
# # model(img)

# # dataset settings
# dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path, 'rgb_dir':args.rgb_dir, 'depth_dir':args.depth_dir}
# dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

# val_dataset = get_dataset(**dataset_kwargs, is_train=False)
# # sampler_val = torch.utils.data.DistributedSampler(
# #             val_dataset, num_replicas=utils.get_world_size(), shuffle=False)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
#                                              pin_memory=True)

# results_dict = validate_single(val_loader,model,device=device, args=args,lowGPU=True)
# print(results_dict)