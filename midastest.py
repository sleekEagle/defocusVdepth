import torch
midas_model_type='DPT_BEiT_L_384'

midas=torch.hub.load("intel-isl/MiDaS", midas_model_type,
                               pretrained=True, force_reload=False)



