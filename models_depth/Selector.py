import torch
from torch import nn

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class Selector(nn.Module):
    def __init__(self,blur_model=None,geometry_model=None,freezeblur=True,freezegeo=True):
        super().__init__()
        self.blur_model=blur_model
        self.geometry_model=geometry_model
        if self.blur_model:
            #output of conv_end of AENet: torch.Size([bs, 16, 480, 480])
            self.outputs_blur={}
            self.blur_model.conv_end.register_forward_hook(get_activation("conv_end",self.outputs_blur))
            blur_model_params = self.blur_model.parameters()
            if freezeblur:
                for p in blur_model_params:
                    p.requires_grad = False
            self.blur_bridge=nn.Sequential(
                nn.Conv2d(16, 16, 3, stride=1, padding=1),
                nn.ReLU()
            )
        
        if self.geometry_model:
            #output of decoder of VPD model: torch.Size([bs, 192, 480, 480])
            self.outputs_geo={}
            self.geometry_model.decoder.register_forward_hook(get_activation("decoder",self.outputs_geo))
            geometry_model_params = self.geometry_model.parameters()
            if freezegeo:
                for p in geometry_model_params:
                    p.requires_grad = False
            self.geo_bridge=nn.Sequential(
                nn.Conv2d(192, 16, 3, stride=1, padding=1),
                nn.ReLU()
            )
        
        if self.blur_model and not self.geometry_model:
            self.conv_selector=nn.Sequential(
                nn.Conv2d(16, 16, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 3, stride=1, padding=1),
            )


    def forward(self,input_RGB,class_id):
        with torch.no_grad():
            if self.blur_model:
                blur_depth,_=self.blur_model(input_RGB,flag_step2=True)
                blur_features=self.blur_bridge(self.outputs_blur['conv_end'])
            if self.geometry_model:
                geometry_depth=self.geometry_model(input_RGB,class_ids=class_id)['pred_d']
                geo_features=self.geo_bridge(self.outputs_geo['decoder'])
            #blur_depth:torch.Size([4, 1, 480, 480])
            #geo_depth:torch.Size([4, 1, 480, 480])
        if self.blur_model and not self.geometry_model:
            selector_feat=blur_features
            weights = self.conv_selector(selector_feat)

        # selector_feat=torch.cat((blur_features,geo_features),dim=1)
        
        #selector_feat:torch.Size([bs, 32, 480, 480])
        
        # d_pred_cat=torch.cat((blur_depth,geometry_depth),dim=1)
        # final_depth=torch.unsqueeze(torch.sum(d_pred_cat*weights,dim=1),dim=1)
        return weights
