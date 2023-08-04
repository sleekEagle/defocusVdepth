import torch
from torch import nn

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class Selector(nn.Module):
    def __init__(self,blur_model=None,geometry_model=None):
        super().__init__()
        self.blur_model=blur_model
        self.geometry_model=geometry_model

    def forward(self,input_RGB,class_id):
        with torch.no_grad():            
            if self.blur_model:
                pred_blur,_=self.blur_model(input_RGB)
            if self.geometry_model:
                pred_geo = self.geometry_model(input_RGB, class_ids=class_id)
                pred_geo=pred_geo['pred_d']
            #blur_depth:torch.Size([bs, 1, 480, 480])
            #geo_depth:torch.Size([bs, 1, 480, 480])
        if self.blur_model and self.geometry_model:
            to_blur_model=pred_geo<2.0
            pred_geo[to_blur_model]=0.
            pred_blur[~to_blur_model]=0.
            comb_pred=pred_geo+pred_blur
        elif self.blur_model and not self.geometry_model:
            comb_pred=pred_blur
        elif not self.blur_model and self.geometry_model:
            comb_pred=pred_geo
        return comb_pred
