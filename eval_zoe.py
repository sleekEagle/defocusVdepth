import torch
import test
from dataset.base_dataset import get_dataset
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)  # Triggers fresh download of MiDaS repo
from models.zoedepth.models.builder import build_model

@torch.no_grad()
def infer(model,images):
    pred1=model.infer(images)
    pred2=model.infer(torch.flip(images, [3]))
    pred2 = torch.flip(pred2, [3])
    mean_pred = 0.5 * (pred1 + pred2)
    return mean_pred

@torch.no_grad()
def evaluate(model,val_loader,round_vals=True, round_precision=3,conf=None):
    model.eval()
    metrics = test.RunningAverageDict()

    for i, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, depth = sample['image'], sample['depth']
        image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        pred=infer(model,image)
        metrics.update(test.compute_metrics(depth.detach(),pred.detach(),config=conf))

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics

@hydra.main(version_base=None, config_path="configs", config_name="config_local")
def run_eval(conf : DictConfig):
    print(OmegaConf.to_yaml(conf))
    zoe = build_model(conf,'infer')
    zoe=zoe.to(DEVICE)

    # Dataset setting
    dataset_kwargs = {'dataset_name': conf.datasets.nyudepthv2.dataset, 'data_path': conf.datasets.nyudepthv2.data_path,'rgb_dir':conf.datasets.nyudepthv2.rgb_dir, 'depth_dir':conf.datasets.nyudepthv2.depth_dir}
    dataset_kwargs['crop_size'] = (conf.models[conf.common.train.image_model].train.input_height, conf.models[conf.common.train.image_model].train.input_width)

    val_dataset = get_dataset(**dataset_kwargs, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                            num_workers=0,pin_memory=True)

    metrics=evaluate(zoe,val_loader,conf=conf)
    print("metrics",metrics)



if __name__ == "__main__":
    run_eval()

# # From URL
# from models_depth.zoedepth.utils.misc import get_image_from_url

# # Example URL
# URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4W8H_Nxk_rs3Vje_zj6mglPOH7bnPhQitBH8WkqjlqQVotdtDEG37BsnGofME3_u6lDk&usqp=CAU"


# from PIL import Image
# import torchvision.transforms as transforms
# image = get_image_from_url(URL)  # fetch
# depth = zoe.infer_pil(image)
# transform = transforms.Compose([transforms.PILToTensor()])
# img_tensor = transform(image)


# import utils
# utils.show_torch_images_jux(img_tensor,torch.from_numpy(depth))





