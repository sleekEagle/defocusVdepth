# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import cv2
from dataset.base_dataset import BaseDataset
import json
import scipy


def get_blur(s1,s2):
    s2[s2==0]=-1
    blur=abs(s2-s1)/s2
    return blur

class nyudepthv2(BaseDataset):
    def __init__(self, data_path, rgb_dir,depth_dir,filenames_path='./dataset/filenames/',
                 is_train=True, crop_size=(448, 576), scale_size=None):
        super().__init__(crop_size)


        if crop_size[0] > 480:
            scale_size = (int(crop_size[0]*640/480), crop_size[0])

        self.scale_size = scale_size

        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'nyu_depth_v2')
        self.rgbpath=os.path.join(self.data_path,rgb_dir)
        self.depthpath=os.path.join(self.data_path,depth_dir)
        
        #read scene names
        scene_path=os.path.join(self.data_path, 'scenes.mat')
        self.scenes=scipy.io.loadmat(scene_path)['scenes']

        #read splits
        splits_path=os.path.join(self.data_path, 'splits.mat')
        splits=scipy.io.loadmat(splits_path)
        if is_train:
            self.file_idx=list(splits['trainNdxs'][:,0])
        else:
            self.file_idx=list(splits['testNdxs'][:,0])

        self.image_path_list = []
        self.depth_path_list = []

        with open('nyu_class_list.json', 'r') as f:
            self.class_list = json.load(f)
 
        phase = 'train' if is_train else 'test'
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.file_idx)))

    def __len__(self):
        return len(self.file_idx)

    def __getitem__(self, idx):
        # num=int(self.filenames_list[idx].split(' ')[0].split('/')[-1].split('.')[-2].split('_')[-1])
        # img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        num=self.file_idx[idx]
        # gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        gt_path=os.path.join(self.depthpath,(str(num)+".png"))
        img_path=os.path.join(self.rgbpath,(str(num)+".png"))
        # filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]
        scene_name=self.scenes[num-1][0][0][:-5]

        class_id = -1
        for i, name in enumerate(self.class_list):
            if name in scene_name:
                class_id = i
                break

        assert class_id >= 0
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
        blur = get_blur(0.1,depth)

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(depth, (self.scale_size[0], self.scale_size[1]))
        
        if self.is_train:
            image,depth,blur = self.augment_training_data(image, depth,blur)
        else:
            image,depth,blur = self.augment_test_data(image, depth,blur)

        depth = depth / 1000.0  # convert in meters

        return {'image': image, 'depth': depth, 'blur':blur, 'class_id': class_id}