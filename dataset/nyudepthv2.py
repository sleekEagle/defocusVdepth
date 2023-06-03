# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import cv2

from dataset.base_dataset import BaseDataset
import json
import h5py
import scipy 

class nyudepthv2(BaseDataset):
    def __init__(self, data_path,rgb_dir,depth_dir,
                 is_train=True, crop_size=(448, 576), scale_size=None):
        super().__init__(crop_size)


        if crop_size[0] > 480:
            scale_size = (int(crop_size[0]*640/480), crop_size[0])

        self.scale_size = scale_size

        self.is_train = is_train

        self.image_path_list = []
        self.depth_path_list = []

        with open('nyu_class_list.json', 'r') as f:
            self.class_list = json.load(f)

        #read scene names
        scene_path=os.path.join(data_path, 'scenes.mat')
        self.scenes=scipy.io.loadmat(scene_path)['scenes']

        #read splits
        splits_path=os.path.join(data_path, 'splits.mat')
        splits=scipy.io.loadmat(splits_path)
        if is_train:
            self.file_idx=list(splits['trainNdxs'][:,0])
        else:
            self.file_idx=list(splits['testNdxs'][:,0])

        self.rgbpath=os.path.join(data_path,rgb_dir)
        self.depthpath=os.path.join(data_path,depth_dir)
  
        phase = 'train' if is_train else 'test'
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.file_idx)))

    def __len__(self):
        return len(self.file_idx)

    def __getitem__(self, idx):
        img_path=os.path.join(self.rgbpath,str(self.file_idx[idx])+'.png')
        depth_path=os.path.join(self.depthpath,str(self.file_idx[idx])+'.png')
        print('img_path:'+str(img_path))
        print('depth_path:'+str(depth_path))
        scene_name=self.scenes[idx][0][0][:-5]
        print("scene:"+str(scene_name))

        class_id = -1
        for i, name in enumerate(self.class_list):
            if name in scene_name:
                class_id = i
                break
        print('class id:'+str(class_id))
        
        assert class_id >= 0
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')

        # print(image.shape, depth.shape, self.scale_size)

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(depth, (self.scale_size[0], self.scale_size[1]))
        
        # print(image.shape, depth.shape, self.scale_size)

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        depth = depth / 1000.0  # convert in meters

        return {'image': image, 'depth': depth, 'filename': scene_name, 'class_id': class_id}
