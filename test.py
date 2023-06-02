import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
sys.path.append('C:\\Users\\lahir\\code\\defocusVdepth\\stable-diffusion\\')
from models_depth.model import VPDDepth
import utils_depth.metrics as metrics
import utils_depth.logging as logging

from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions
import utils

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

def validate(val_loader, model, device, args):
    
    if args.save_eval_pngs or args.save_visualize:
        result_path = os.path.join(args.result_dir, args.exp_name)
        # if args.rank == 0:
        #     logging.check_and_make_dirs(result_path)
        print("Saving result images in to %s" % result_path)

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
        filename = batch['filename'][0]
        class_id = batch['class_id']

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
            
            #loop insted of sending through the network for lower memory GPUs (but slower)
            num=input_RGB.shape[0]
            predlist=torch.empty((0,1,input_RGB.shape[-2],input_RGB.shape[-1])).to(device)
            for i in range(num):
                img=torch.unsqueeze(input_RGB[i,:,:,:],dim=0)
                c=torch.unsqueeze(class_ids[i],dim=0)
                pred = model(img, class_ids=c)
                pred_d = pred['pred_d']
                blur=pred['blur']
                predlist=torch.cat((predlist,pred_d),dim=0)

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
    
        if args.save_eval_pngs:
            save_path = os.path.join(result_path, filename)
            if save_path.split('.')[-1] == 'jpg':
                save_path = save_path.replace('jpg', 'png')
            pred_d = pred_d.squeeze()
            if args.dataset == 'nyudepthv2':
                pred_d = pred_d.cpu().numpy() * 1000.0
                cv2.imwrite(save_path, pred_d.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                pred_d = pred_d.cpu().numpy() * 256.0
                cv2.imwrite(save_path, pred_d.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
        if args.save_visualize:
            save_path = os.path.join(result_path, filename)
            pred_d_numpy = pred_d.squeeze().cpu().numpy()
            pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
            pred_d_numpy = pred_d_numpy.astype(np.uint8)
            pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
            cv2.imwrite(save_path, pred_d_color)

        ddp_logger.update(**computed_result)
        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    ddp_logger.synchronize_between_processes()

    for key in result_metrics.keys():
        result_metrics[key] = ddp_logger.meters[key].global_avg

    return result_metrics

opt = TestOptions()
args = opt.initialize().parse_args()
args.shift_window_test=True
args.flip_test=True
print(args)

device = torch.device(0)

model = VPDDepth(args=args)

cudnn.benchmark = True
model.to(device)
from collections import OrderedDict
model_weight = torch.load(args.ckpt_dir)['model']
if 'module' in next(iter(model_weight.items()))[0]:
    model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
model.load_state_dict(model_weight, strict=False)

model.eval()

# img=torch.rand((1,3,480,480))
# img=img.to(device)
# model(img)

# dataset settings
dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

val_dataset = get_dataset(**dataset_kwargs, is_train=False)
# sampler_val = torch.utils.data.DistributedSampler(
#             val_dataset, num_replicas=utils.get_world_size(), shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                             pin_memory=True)

results_dict = validate(val_loader,model,device=device, args=args)
# torch.utils.data.get_worker_info() 


# for batch_idx, batch in enumerate(val_loader):
#     input_RGB = batch['image'].to(device)
#     depth_gt = batch['depth'].to(device)
#     filename = batch['filename'][0]
#     class_id = batch['class_id']
#     break


# with torch.no_grad():
#     if args.shift_window_test:
#         bs, _, h, w = input_RGB.shape
#         assert w > h and bs == 1
#         interval_all = w - h
#         interval = interval_all // (args.shift_size-1)
#         sliding_images = []
#         sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
#         class_ids = []
#         for i in range(args.shift_size):
#             sliding_images.append(input_RGB[..., :, i*interval:i*interval+h])
#             sliding_masks[..., :, i*interval:i*interval+h] += 1
#             class_ids.append(class_id)
#         input_RGB = torch.cat(sliding_images, dim=0)
#         class_ids = torch.cat(class_ids, dim=0)
#     if args.flip_test:
#         input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
#         class_ids = torch.cat((class_ids, class_ids), dim=0)
#     pred = model(input_RGB, class_ids=class_ids)
# pred_d = pred['pred_d']

# import matplotlib.pyplot as plt

# input_RGB=input_RGB[:,:,:,:480]
# out=model(input_RGB,class_id)
# dpred=out['pred_d']

# p=dpred.detach().cpu()[0,0,:,:].numpy()

# plt.imshow(p)
# plt.show()

# dgt=depth_gt.cpu()[0,:,:].numpy()
# plt.imshow(dgt)
# plt.show()

# np.min(p),np.max(p),np.mean(p)
# np.min(dgt),np.max(dgt),np.mean(dgt)











# import os
# path='C:\\Users\\lahir\\code\\VPD\\depth\\nyu_depth_v2\\sync\\sync\\'
# path='C:\\Users\\lahir\\data\\VPD\\nyu_depth_v2\\official_splits\\train\\'


# filelist = []
# for root, dirs, files in os.walk(path):
# 	for file in files:
# 		if(file[-3:]=="jpg"):
# 		    filelist.append(file)
		


