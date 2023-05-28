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

opt = TestOptions()
args = opt.initialize().parse_args()
print(args)
device = torch.device(0)

model = VPDDepth(args=args)

cudnn.benchmark = True
model.to(device)
from collections import OrderedDict
model_weight = torch.load(args.ckpt_dir)['model']


#dataset settings
dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

val_dataset = get_dataset(**dataset_kwargs, is_train=False)

