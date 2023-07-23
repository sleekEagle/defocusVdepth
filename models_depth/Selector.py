import torch
from torch import nn

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class Selector(nn.Module):
    def __init__(self,blur_model,geometry_model):
        super().__init__()
        self.blur_model=blur_model
        self.geometry_model=geometry_model
        self.outputs_blur={}
        self.blur_model.conv_end.register_forward_hook(get_activation("conv_end",self.outputs_blur))
        #output of conv_end: torch.Size([bs, 16, 480, 480])
        self.outputs_geo={}
        self.geometry_model.decoder.register_forward_hook(get_activation("decoder",self.outputs_geo))
        #output of decoder: torch.Size([bs, 192, 480, 480])

        self.conv_selector=nn.Sequential(
            nn.Conv2d(208, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 3, stride=1, padding=1),
            nn.Softmax(dim=1)
        )

        #freeze all layers of both models
        geometry_model_params = self.geometry_model.parameters()
        blur_model_params = self.blur_model.parameters()
        for p in geometry_model_params:
            p.requires_grad = False
        # for p in blur_model_params:
        #     p.requires_grad = False

    def forward(self,input_RGB,class_id):
        with torch.no_grad():
            blur_depth,_=self.blur_model(input_RGB,flag_step2=True)
            geometry_depth=self.geometry_model(input_RGB,class_ids=class_id)['pred_d']
            #blur_depth:torch.Size([4, 1, 480, 480])
            #geo_depth:torch.Size([4, 1, 480, 480])
        #obtainig features
        selector_feat=torch.cat((self.outputs_blur['conv_end'],self.outputs_geo['decoder']),dim=1)
        #selector_feat :torch.Size([bs, 208, 480, 480])
        weights = self.conv_selector(selector_feat)
        d_pred_cat=torch.cat((blur_depth,geometry_depth),dim=1)
        final_depth=torch.unsqueeze(torch.sum(d_pred_cat*weights,dim=1),dim=1)
        return final_depth