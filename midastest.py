import torch
import cv2
midas_model_type='DPT_BEiT_L_384'

midas=torch.hub.load("intel-isl/MiDaS", midas_model_type,
                               pretrained=True, force_reload=False)





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




