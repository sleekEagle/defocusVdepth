# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from models.trainers.loss import GradL1Loss, SILogLoss
from models.zoedepth.utils.config import DATASETS_CONFIG
from models.zoedepth.utils.misc import compute_metrics
from models.zoedepth.data.preprocess import get_black_border

from .base_trainer import BaseTrainer
from torchvision import transforms
from PIL import Image
import numpy as np

#read dataset config from config file

class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device,model_name=config.models[config.common.model_name].model.name)
        self.device = device
        self.mseloss=nn.MSELoss()
        self.scaler = amp.GradScaler(enabled=self.config.models[self.config.common.model_name].train.use_amp)

    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w
        batch["depth"].shape : batch_size, 1, h, w
        """

        images, depths_gt = batch['image'].to(
            self.device), batch['depth'].to(self.device)
        dataset = batch['dataset'][0]
        dblur_gt=batch['dblur'].to(self.device)
        blur_gt=batch['blur'].to(self.device)

        b, c, h, w = images.size()
        mask = batch["mask"].to(self.device).to(torch.bool)

        losses = {}

        with amp.autocast(enabled=self.config.models[self.config.common.model_name].train.use_amp):

            pred_blur,pred_dblur = self.model(images)

            l_blur = self.mseloss(blur_gt[mask],pred_blur[mask])
            l_dblur = self.mseloss(dblur_gt[mask],pred_dblur[mask])

            loss = self.config.models.blurnet.train.blur_w*l_blur + self.config.models.blurnet.train.dblur_w*l_dblur
            losses['blur loss'] = l_blur
            losses['dblur loss'] = l_dblur


        self.scaler.scale(loss).backward()

        if self.config.models[self.config.common.model_name].train.clip_grad:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.models.zoedepth.train.clip_grad)

        self.scaler.step(self.optimizer)

        if self.should_log and (self.step % int(self.config.common.train.log_images_every * self.iters_per_epoch)) == 0:
            # -99 is treated as invalid depth in the log_images function and is colored grey.
            depths_gt[torch.logical_not(mask)] = -99
            
            # print('accessing config...****************')
            # print(self.config.datasets.nyudepthv2.min_depth)
            # print('*********************************')

            self.log_images(rgb={"Input": images[0, ...]}, depth={"GTBlur": blur_gt[0], "PredictedBlur": pred_blur[0]}, prefix="Train",
                            min_depth=self.config.datasets.nyudepthv2.min_depth, max_depth=self.config.datasets.nyudepthv2.max_depth)


        self.scaler.update()
        self.optimizer.zero_grad()

        return losses
    
    @torch.no_grad()
    def eval_infer(self, x):
        en=False if self.config.models[self.config.common.model_name].train.use_amp==0 else 1
        with amp.autocast(enabled=en):
            m = self.model.module if self.config.common.train.multigpu else self.model
            pred_blur,pred_dblur = m(x)
        return pred_blur,pred_dblur

    @torch.no_grad()
    def crop_aware_infer(self, x):
        # if we are not avoiding the black border, we can just use the normal inference
        if not self.config.get("avoid_boundary", False):
            return self.eval_infer(x)
        
        # otherwise, we need to crop the image to avoid the black border
        # For now, this may be a bit slow due to converting to numpy and back
        # We assume no normalization is done on the input image

        # get the black border
        assert x.shape[0] == 1, "Only batch size 1 is supported for now"
        x_pil = transforms.ToPILImage()(x[0].cpu())
        x_np = np.array(x_pil, dtype=np.uint8)
        black_border_params = get_black_border(x_np)
        top, bottom, left, right = black_border_params.top, black_border_params.bottom, black_border_params.left, black_border_params.right
        x_np_cropped = x_np[top:bottom, left:right, :]
        x_cropped = transforms.ToTensor()(Image.fromarray(x_np_cropped))

        # run inference on the cropped image
        pred_depths_cropped = self.eval_infer(x_cropped.unsqueeze(0).to(self.device))

        # resize the prediction to x_np_cropped's size
        pred_depths_cropped = nn.functional.interpolate(
            pred_depths_cropped, size=(x_np_cropped.shape[0], x_np_cropped.shape[1]), mode="bilinear", align_corners=False)
        

        # pad the prediction back to the original size
        pred_depths = torch.zeros((1, 1, x_np.shape[0], x_np.shape[1]), device=pred_depths_cropped.device, dtype=pred_depths_cropped.dtype)
        pred_depths[:, :, top:bottom, left:right] = pred_depths_cropped

        return pred_depths



    def validate_on_batch(self, batch, val_step):
        images = batch['image'].to(self.device)
        depths_gt = batch['depth'].to(self.device)
        dataset = batch['dataset'][0]
        mask = batch["mask"].to(self.device)
        dblur_gt=batch['dblur'].to(self.device)
        blur_gt=batch['blur'].to(self.device)

        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth']:
                return None, None

        depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
        mask = mask.squeeze().unsqueeze(0)
        if dataset == 'nyu':
            pred_depths = self.crop_aware_infer(images)
        else:
            pred_blurs,pred_dblurs = self.eval_infer(images)
        pred_blurs = pred_blurs.squeeze().unsqueeze(0)
        pred_dblurs = pred_dblurs.squeeze().unsqueeze(0)

        with amp.autocast(enabled=self.config.models.zoedepth.train.use_amp):
            l_blur = self.mseloss(blur_gt[mask],pred_blurs[mask])
            l_dblur = self.mseloss(dblur_gt[mask],pred_dblurs[mask])

        blur_metrics = compute_metrics(blur_gt, pred_blurs, **self.config)
        dblur_metrics = compute_metrics(blur_gt, pred_blurs, **self.config)
        losses = {f"MSE blur loss": l_blur.item(),f"MSE dblur loss": l_dblur.item()}

        if val_step == 1 and self.should_log:
            mask=mask.unsqueeze(0)
            depths_gt[torch.logical_not(mask)] = -99
            self.log_images(rgb={"Input": images[0]}, depth={"GT_Blur": blur_gt[0], "Predicted_Blur": pred_blurs[0],"GT_DBlur": dblur_gt[0], "Predicted_DBlur": pred_dblurs[0]}, prefix="Test",
                            min_depth=self.config.datasets.nyudepthv2.min_depth, max_depth=self.config.datasets.nyudepthv2.max_depth)

        return blur_metrics, losses
