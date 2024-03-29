import numpy as np
import torch
import torch.nn
import torch.backends.cudnn as cudnn
import sys
sys.path.append('stable-diffusion')
from models.model import VPDDepth
import utils_depth.metrics as metrics
import utils_depth.logging as logging

from dataset.base_dataset import get_dataset
import utils
import math
from utils_depth.criterion import SiLogLoss


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


def validate_dist_2d(val_loader, model, criterion_d, device_id, args,min_dist=0.0,max_dist=10.0,model_name=None,lowGPU=False,crop=256):

    if device_id == 0:
        depth_loss = logging.AverageMeter()
    model.eval()


    ddp_logger = utils.MetricLogger()
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(val_loader):
        # if batch_idx%10==0:
        #     print(str(round(batch_idx/len(val_loader),2))+' done.')
        input_RGB = batch['image'].to(device_id)
        depth_gt = batch['depth'].to(device_id)
        class_id = batch['class_id']

        with torch.no_grad():
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert h>crop and w>crop and bs == 1
                interval_all_x=w-crop
                interval_all_y=h-crop
                interval_x=interval_all_x//(args.shift_size-1)
                interval_y=interval_all_y//(args.shift_size-1)

                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
                class_ids = []
                for i in range(args.shift_size):
                    for j in range(args.shift_size):
                        sliding_images.append(input_RGB[..., j*interval_y:j*interval_y+crop, i*interval_x:i*interval_x+crop])
                        sliding_masks[..., :, j*interval_y:j*interval_y+crop, i*interval_x:i*interval_x+crop] += 1
                        class_ids.append(class_id)
                input_RGB = torch.cat(sliding_images, dim=0)
                class_ids = torch.cat(class_ids, dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
                class_ids = torch.cat((class_ids, class_ids), dim=0)
            
            #loop insted of sending through the network for lower memory GPUs (this is slower)
            if(lowGPU):
                num=input_RGB.shape[0]
                predlist=torch.empty((0,1,input_RGB.shape[-2],input_RGB.shape[-1])).to(input_RGB.device)
                for i in range(num):
                    img=torch.unsqueeze(input_RGB[i,:,:,:],dim=0)
                    c=torch.unsqueeze(class_ids[i],dim=0)
                    if model_name=='defnet':
                        pred_d,pred_b= model(img)
                    elif model_name=='midas':
                        pred_d=model(img)
                        pred_d=torch.unsqueeze(pred_d,dim=1)
                    elif model_name=='vpd':
                        pred = model(img, class_ids=c)
                        pred_d = pred['pred_d']
                    elif model_name=='combined':
                        pred_d=model(img, class_id=c)
                    else:
                        return -1
                    predlist=torch.cat((predlist,pred_d),dim=0)
                    pred_d=predlist
            else:
                if model_name=='defnet':
                    pred_d,_= model(input_RGB)
                elif model_name=='zoe':
                    pred_d=model.infer(input_RGB)
                elif model_name=='midas':
                    pred_d=model(input_RGB)
                    pred_d=torch.unsqueeze(pred_d,dim=1)
                elif model_name=='vpd':
                    pred = model(input_RGB, class_ids=class_ids)
                    pred_d = pred['pred_d']
                elif model_name=='combined':
                    pred_d,_=model(input_RGB,class_ids)
                else:
                    return -1
        if args.flip_test:
            batch_s = pred_d.shape[0]//2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0
        if args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(args.shift_size):
                for j in range(args.shift_size):
                    pred_s[..., :, j*interval_y:j*interval_y+crop, i*interval_x:i*interval_x+crop] += pred_d[i*j:i*j+1]
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

#provides distance wise error
def validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=0.0,max_dist=10.0,model_name=None,lowGPU=False):

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
        with torch.no_grad():
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert w>h and bs == 1
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
                predlist=torch.empty((0,1,input_RGB.shape[-2],input_RGB.shape[-1])).to(input_RGB.device)
                for i in range(num):
                    img=torch.unsqueeze(input_RGB[i,:,:,:],dim=0)
                    c=torch.unsqueeze(class_ids[i],dim=0)
                    if model_name=='defnet':
                        pred_d,pred_b= model(img)
                    elif model_name=='midas':
                        pred_d=model(img)
                        pred_d=torch.unsqueeze(pred_d,dim=1)
                    elif model_name=='vpd':
                        pred = model(img, class_ids=c)
                        pred_d = pred['pred_d']
                    elif model_name=='combined':
                        pred_d=model(img, class_id=c)
                    else:
                        return -1
                    predlist=torch.cat((predlist,pred_d),dim=0)
                    pred_d=predlist
            else:
                if model_name=='defnet':
                    pred_d,_= model(input_RGB)
                elif model_name=='midas':
                    pred_d=model(input_RGB)
                    pred_d=torch.unsqueeze(pred_d,dim=1)
                elif model_name=='vpd':
                    pred = model(input_RGB, class_ids=class_ids)
                    pred_d = pred['pred_d']
                elif model_name=='combined':
                    pred_d,_=model(input_RGB,class_ids)
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


criterion_d = SiLogLoss()
def vali_dist(val_loader,model,device_id,args,logger,model_name):
        logger.info('testingt....')
        results_dict,loss_d=validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=0.0,max_dist=1.0,model_name=model_name)
        print("dist : 0-1 " + str(results_dict))
        logger.info("dist : 0-1 " + str(results_dict))

        results_dict,loss_d=validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=1.0,max_dist=2.0,model_name=model_name)
        print("dist : 1-2 " + str(results_dict))
        logger.info("dist : 1-2 " + str(results_dict))

        results_dict,loss_d=validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=2.0,max_dist=3.0,model_name=model_name)
        print("dist : 2-3 " + str(results_dict))
        logger.info("dist : 2-3 " + str(results_dict))

        results_dict,loss_d=validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=3.0,max_dist=4.0,model_name=model_name)
        print("dist : 3-4 " + str(results_dict))
        logger.info("dist : 3-4 " + str(results_dict))

        results_dict,loss_d=validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=4.0,max_dist=5.0,model_name=model_name)
        print("dist : 4-5 " + str(results_dict))
        logger.info("dist : 4-5 " + str(results_dict))

        results_dict,loss_d=validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=5.0,max_dist=6.0,model_name=model_name)
        print("dist : 5-6 " + str(results_dict))
        logger.info("dist : 5-6 " + str(results_dict))

        results_dict,loss_d=validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=6.0,max_dist=8.0,model_name=model_name)
        print("dist : 6-8 " + str(results_dict))
        logger.info("dist : 6-8 " + str(results_dict))
        results_dict,loss_d=validate_dist(val_loader, model, criterion_d, device_id, args,min_dist=8.0,max_dist=10.0,model_name=model_name)
        print("dist : 8-10 " + str(results_dict))
        logger.info("dist : 8-10 " + str(results_dict))


'''
Evaluating Zoedepth model
'''
class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}

def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

    
def compute_metrics(gt, pred, interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, **kwargs):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """
    if 'config' in kwargs:
        config = kwargs['config']
        garg_crop = config.datasets.nyudepthv2.garg_crop
        eigen_crop = config.datasets.nyudepthv2.eigen_crop
        min_depth_eval = config.common.eval.min_depth
        max_depth_eval = config.common.eval.max_depth

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)

    pred = pred.squeeze().cpu().numpy()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze().cpu().numpy()
    valid_mask = np.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
    valid_mask = np.logical_and(valid_mask, eval_mask)
    return compute_errors(gt_depth[valid_mask], pred[valid_mask])

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
