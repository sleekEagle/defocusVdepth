import torch.nn as nn
import torch.nn.functional as F
import torch

def get_activation(name,bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class Midas(nn.Module):
    def __init__(self,layers=['l4_rn', 'r4', 'r3', 'r2', 'r1'],model_type='DPT_SwinV2_L_384'):
        super().__init__()
        self.layers=layers
        self.activation={}
        self.handles=[]
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.attach_hooks()

        self.bridge = nn.Sequential(
            nn.Conv2d(256*len(self.layers), 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(33, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    '''
    outputs and shapes:
    out_conv : bs,32,in,in
    l4_rn    : bs,256,in/32,in/32
    r4       : bs,256,in/16,in/16
    r3       : bs,256,in/8,in/8
    r2       : bs,256,in/4,in/4
    r1       : bs,256,in/2,in/2
    '''
    def attach_hooks(self):
        if "out_conv" in self.layers:
            self.handles.append(list(self.midas.scratch.output_conv.children())[3].register_forward_hook(get_activation("out_conv", self.activation)))
        if "r4" in self.layers:
            self.handles.append(self.midas.scratch.refinenet4.register_forward_hook(get_activation("r4", self.activation)))
        if "r3" in self.layers:
            self.handles.append(self.midas.scratch.refinenet3.register_forward_hook(get_activation("r3", self.activation)))
        if "r2" in self.layers:
            self.handles.append(self.midas.scratch.refinenet2.register_forward_hook(get_activation("r2", self.activation)))
        if "r1" in self.layers:
            self.handles.append(self.midas.scratch.refinenet1.register_forward_hook(get_activation("r1", self.activation)))
        if "l4_rn" in self.layers:
            self.handles.append(self.midas.scratch.layer4_rn.register_forward_hook(get_activation("l4_rn", self.activation)))

    def forward(self, x):
        bs,_,h,w=x.shape
        blur=self.midas(x)
        blur=torch.unsqueeze(blur,dim=1)
        features = [F.interpolate(self.activation[k],[h,w],mode='bilinear') for k in self.layers]
        features_cat=torch.cat(features,dim=1)
        f=self.bridge(features_cat)
        fb=torch.cat((f,blur),dim=1)
        depth=self.depth_conv(fb)
        return depth,blur
