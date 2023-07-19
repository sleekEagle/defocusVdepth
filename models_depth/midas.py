import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class MidasCore(nn.Module):
    def __init__(self, midas, trainable=False, fetch_features=True, layer_names=('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1'), freeze_bn=False, keep_aspect_ratio=True,
                 img_size_in=480,img_size_out=480):
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

        self.inter1 = Interpolate(size=(int(img_size_in/2),int(img_size_in/2)), mode='bilinear')
        self.inter2 = Interpolate(size=(int(img_size_in),int(img_size_in)), mode='bilinear')
        self.feature_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        self.feature_conv2 = nn.Sequential(
            nn.Conv2d(80, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(17, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.img_size_in=img_size_in
        self.img_size_out=img_size_out

        if self.img_size_out>self.img_size_in:
            #downscale the inputs to the network
            self.initial_downscale=Interpolate(size=(self.img_size_in,self.img_size_in), mode='bilinear')
            #we need to scale up the outputs of the network
            self.final_upscale=Interpolate(size=(self.img_size_out,self.img_size_out), mode='bilinear')

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
        '''
        layer_names (in order)=('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1')
        shapes:
        out_conv:torch.Size([5, 32, 480, 480])
        l4_rn:torch.Size([5, 256, 15, 15])
        r4:torch.Size([5, 256, 30, 30])
        r3:torch.Size([5, 256, 60, 60])
        r2:torch.Size([5, 256, 120, 120])
        r1:torch.Size([5, 256, 240, 240])
        '''

        with torch.set_grad_enabled(self.trainable):
            #downscale input image if needed
            if self.img_size_out>self.img_size_in:
                print('before downscale:'+str(x.shape))
                x=self.initial_downscale(x)
                print('after downscale:'+str(x.shape))
            blur = self.core(x)

        out = [self.core_out[k] for k in self.layer_names]
        #upsample the decoder and bottleneck features
        l4_rn_=self.feature_conv1(out[1])
        l4_rn_interp=self.inter1(l4_rn_)

        r4_=self.feature_conv1(out[2])
        r4_interp=self.inter1(r4_)
       
        r3_=self.feature_conv1(out[3])
        r3_interp=self.inter1(r3_)
        
        r2_=self.feature_conv1(out[4])
        r2_interp=self.inter1(r2_)
        
        r1_=self.feature_conv1(out[5])
        r1_interp=r1_

        #concat all the outputs
        feat1=torch.cat((l4_rn_interp,r4_interp,r3_interp,r2_interp,r1_interp),dim=1)
        feat2_=self.feature_conv2(feat1)
        feat2=self.inter2(feat2_)

        #concat with the blur
        blur_=torch.unsqueeze(blur,dim=1)
        feat3=torch.cat((blur_,feat2),dim=1)
        depth_=self.depth_conv(feat3)
        depth=torch.squeeze(depth_,dim=1)

        if self.img_size_out>self.img_size_in:
            depth=self.final_upscale(torch.unsqueeze(depth,dim=1))
            blur=self.final_upscale(torch.unsqueeze(blur,dim=1))
            depth=torch.squeeze(depth,dim=1)
            blur=torch.squeeze(blur,dim=1)

        return blur,depth,out

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
    def build(midas_model_type="DPT_BEiT_L_384", train_midas=False, use_pretrained_midas=True, fetch_features=False, freeze_bn=True, force_keep_ar=False, force_reload=False,
              img_size_in=384,img_size_out=384):
        midas = torch.hub.load("intel-isl/MiDaS", midas_model_type,
                               pretrained=use_pretrained_midas, force_reload=force_reload)
        print('loaded from hub')
        midas_core = MidasCore(midas, trainable=train_midas, fetch_features=fetch_features,
                               freeze_bn=freeze_bn,img_size_in=img_size_in,img_size_out=img_size_out)
        return midas_core

    @staticmethod
    def build_from_config(config):
        return MidasCore.build(**config)
    
# midas_model_type='DPT_BEiT_L_384'
# use_pretrained_midas=False
# train_midas=True
# freeze_midas_bn=False
# core = MidasCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
#                                train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn,img_size=480)


# import torch
# img=torch.rand((1,3,480,480))
# blur,depth,out=core(img,return_rel_depth=True)

# depth.shape
# for item in out:
#     print(item.shape)





