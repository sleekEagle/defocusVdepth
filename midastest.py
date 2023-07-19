import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class MidasCore(nn.Module):
    def __init__(self, midas, trainable=False, fetch_features=True, layer_names=('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1'), freeze_bn=False, keep_aspect_ratio=True,
                 img_size=384):
        """Midas Base model used for multi-scale feature extraction.

        Args:
            midas (torch.nn.Module): Midas model.
            trainable (bool, optional): Train midas model. Defaults to False.
            fetch_features (bool, optional): Extract multi-scale features. Defaults to True.
            layer_names (tuple, optional): Layers used for feature extraction. Order = (head output features, last layer features, ...decoder features). Defaults to ('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1').
            freeze_bn (bool, optional): Freeze BatchNorm. Generally results in better finetuning performance. Defaults to False.
            keep_aspect_ratio (bool, optional): Keep the aspect ratio of input images while resizing. Defaults to True.
            img_size (int, tuple, optional): Input resolution. Defaults to 384.
        """
        super().__init__()
        self.core = midas
        self.output_channels = None
        self.core_out = {}
        self.trainable = trainable
        self.fetch_features = fetch_features
        # midas.scratch.output_conv = nn.Identity()
        self.handles = []
        # self.layer_names = ['out_conv','l4_rn', 'r4', 'r3', 'r2', 'r1']
        self.layer_names = layer_names

        self.set_trainable(trainable)
        self.set_fetch_features(fetch_features)


        if freeze_bn:
            self.freeze_bn()

    def set_trainable(self, trainable):
        self.trainable = trainable
        if trainable:
            self.unfreeze()
        else:
            self.freeze()
        return self

    def set_fetch_features(self, fetch_features):
        self.fetch_features = fetch_features
        if fetch_features:
            if len(self.handles) == 0:
                self.attach_hooks(self.core)
        else:
            self.remove_hooks()
        return self

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.trainable = False
        return self

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.trainable = True
        return self

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x, return_rel_depth=False):

        with torch.set_grad_enabled(self.trainable):

            # print("Input size to Midascore", x.shape)
            rel_depth = self.core(x)
            # print("Output from midas shape", rel_depth.shape)
            if not self.fetch_features:
                return rel_depth
        out = [self.core_out[k] for k in self.layer_names]

        if return_rel_depth:
            return rel_depth, out
        return out

    def get_rel_pos_params(self):
        for name, p in self.core.pretrained.named_parameters():
            if "relative_position" in name:
                yield p

    def get_enc_params_except_rel_pos(self):
        for name, p in self.core.pretrained.named_parameters():
            if "relative_position" not in name:
                yield p

    def freeze_encoder(self, freeze_rel_pos=False):
        if freeze_rel_pos:
            for p in self.core.pretrained.parameters():
                p.requires_grad = False
        else:
            for p in self.get_enc_params_except_rel_pos():
                p.requires_grad = False
        return self

    def attach_hooks(self, midas):
        if len(self.handles) > 0:
            self.remove_hooks()
        if "out_conv" in self.layer_names:
            self.handles.append(list(midas.scratch.output_conv.children())[
                                3].register_forward_hook(get_activation("out_conv", self.core_out)))
        if "r4" in self.layer_names:
            self.handles.append(midas.scratch.refinenet4.register_forward_hook(
                get_activation("r4", self.core_out)))
        if "r3" in self.layer_names:
            self.handles.append(midas.scratch.refinenet3.register_forward_hook(
                get_activation("r3", self.core_out)))
        if "r2" in self.layer_names:
            self.handles.append(midas.scratch.refinenet2.register_forward_hook(
                get_activation("r2", self.core_out)))
        if "r1" in self.layer_names:
            self.handles.append(midas.scratch.refinenet1.register_forward_hook(
                get_activation("r1", self.core_out)))
        if "l4_rn" in self.layer_names:
            self.handles.append(midas.scratch.layer4_rn.register_forward_hook(
                get_activation("l4_rn", self.core_out)))

        return self

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        return self

    def __del__(self):
        self.remove_hooks()

    @staticmethod
    def build(midas_model_type="DPT_BEiT_L_384", train_midas=False, use_pretrained_midas=True, fetch_features=False, freeze_bn=True, force_keep_ar=False, force_reload=False):
        midas = torch.hub.load("intel-isl/MiDaS", midas_model_type,
                               pretrained=use_pretrained_midas, force_reload=force_reload)
        print('loaded from hub')
        midas_core = MidasCore(midas, trainable=train_midas, fetch_features=fetch_features,
                               freeze_bn=freeze_bn, img_size=384)
        return midas_core

    @staticmethod
    def build_from_config(config):
        return MidasCore.build(**config)
    
midas_model_type='DPT_BEiT_L_384'
use_pretrained_midas=False
train_midas=True
freeze_midas_bn=False
core = MidasCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
                               train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn)



midas=torch.hub.load("intel-isl/MiDaS", midas_model_type,
                               pretrained=True, force_reload=False)

to_tensor = transforms.ToTensor()
img=np.random.random((20,20))*255.0
img_t=to_tensor(img)
np.max(img),torch.max(img_t)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if midas_model_type == "DPT_Large" or midas_model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform



#read image
img = cv2.imread('C:\\Users\\lahir\\Downloads\\fire.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img)
prediction = midas(input_batch)

midas




