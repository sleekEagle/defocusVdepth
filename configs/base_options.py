# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

import argparse
import os
import json
import pathlib
from configs.easydict import EasyDict as edict

ROOT = pathlib.Path(__name__).parent.parent.resolve()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def get_model_config(model_name, model_version=None):
    """Find and parse the .json config file for the model.

    Args:
        model_name (str): name of the model. The config file should be named config_{model_name}[_{model_version}].json under the models/{model_name} directory.
        model_version (str, optional): Specific config version. If specified config_{model_name}_{model_version}.json is searched for and used. Otherwise config_{model_name}.json is used. Defaults to None.

    Returns:
        easydict: the config dictionary for the model.
    """
    config_fname = f"config_{model_name}_{model_version}.json" if model_version is not None else f"config_{model_name}.json"
    config_file = os.path.join(ROOT,"models","zoedepth","models", model_name, config_fname)
    print('conf file',config_file)
    if not os.path.exists(config_file):
        return None

    with open(config_file, "r") as f:
        config = edict(json.load(f))

    return edict(config)

def get_data_config(dataset):
    config_name="config.json"
    config_file = os.path.join(ROOT,"dataset",config_name)
    if not os.path.exists(config_file):
        return None
    with open(config_file, "r") as f:
        config = edict(json.load(f))
    return edict(config)[dataset]


class BaseOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # base configs
        parser.add_argument('--exp_name',   type=str, default='')
        parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')

        #dataset configs
        parser.add_argument('--data_path',    type=str, default=None)
        parser.add_argument('--rgb_dir',    type=str,default=None)
        parser.add_argument('--depth_dir',    type=str,default=None)
        parser.add_argument('--dataset',      type=str,default='nyudepthv2',
                            choices=['nyudepthv2', 'kitti', 'imagepath'])
        parser.add_argument('--batch_size',   type=int, default=None)
        parser.add_argument('--virtual_batch_size',   type=int, default=None)
        parser.add_argument('--workers',      type=int, default=None)
        parser.add_argument('--garg_crop',    type=bool, default=None)
        parser.add_argument('--eigen_crop',    type=bool, default=None)

        #model configs
        parser.add_argument('--blur_model',    type=str, default='defnet')
        parser.add_argument('--midas_type',    type=str, default='DPT_BEiT_L_384')
        parser.add_argument('--image_model',    type=str, default='zoedepth')

        parser.add_argument('--save_dir',    type=str, default='/p/blurdepth/results/defvdep/')
        parser.add_argument('--uid',type=int, default=0)
        parser.add_argument('--tags',type=str, default='')
        parser.add_argument('--project',type=str, default='zoedepth')
        parser.add_argument('--root',type=str, default='.')
        parser.add_argument('--notes',type=str, default='')
        parser.add_argument('--validate_every',type=float, default=0.25)
        parser.add_argument('--log_images_every',type=float, default=0.1)
        parser.add_argument('--prefetch',type=bool, default=False)

        # depth configs
        parser.add_argument('--max_depth',      type=float, default=10.0)
        parser.add_argument('--max_depth_eval', type=float, default=10.0)
        parser.add_argument('--min_depth_eval', type=float, default=1e-3)        
        parser.add_argument('--do_kb_crop',     type=int, default=1)
        parser.add_argument('--kitti_crop', type=str, default=None,
                            choices=['garg_crop', 'eigen_crop'])
        
        #how many filters are used to preict depth out of 192
        parser.add_argument('--blur_n',     type=int, default=20)
        parser.add_argument('--method',     type=int, default=0)

        parser.add_argument('--pretrained',    type=str, default='')
        parser.add_argument('--drop_path_rate',     type=float, default=0.3)
        parser.add_argument('--use_checkpoint',   type=str2bool, default='False')
        parser.add_argument('--num_deconv',     type=int, default=3)
        parser.add_argument('--num_filters', nargs='+', type=int, default=[32,32,32])
        parser.add_argument('--deconv_kernels', nargs='+', type=int, default=[2,2,2])

        parser.add_argument('--shift_window_test', action='store_true')     
        parser.add_argument('--shift_size',  type=int, default=2)
        parser.add_argument('--flip_test', action='store_true')       
        
        return parser
    

    def get_arg_dict(self):
        conf={}
        args=vars(self.initialize().parse_args())
        #read model config files from json
        conf_model=get_model_config(args['image_model'])
        #read dataset config file form json
        conf_data=get_data_config(args['dataset'])

        conf.update(conf_model)
        conf.update(conf_data)
        #replace conf with options from args if there are matches
        conf.update((k,v) for k,v in args.items() if v is not None)
        # return edict(conf)
        return args
    

opt = BaseOptions()
config=opt.get_arg_dict()
print(config)

config['image_model']

get_model_config(config['image_model'])

