#!/usr/bin/env python
# -*- coding: utf-8 -*-
# copied from https://github.com/cleinc/bts with some modifications

from __future__ import print_function

import h5py
import numpy as np
import os
import scipy.io
import sys
import cv2
from pathlib import Path


def convert_image(i,filledDepth,depth_raw,image):

    idx = int(i) + 1
    # if idx in train_images:
    #     train_test = "train"
    # else:
    #     assert idx in test_images, "index %d neither found in training set nor in test set" % idx
    #     train_test = "test"

    # folder = "%s/%s" % (out_folder, train_test)
    folder=out_folder

    #save filled depth image
    img_depth = filledDepth * 1000.0
    img_depth_uint16 = img_depth.astype(np.uint16)
    depth_path=(folder / "filledDepth")
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    depth_path=depth_path / (str(idx) +".png")
    # img_depth_uint16=img_depth_uint16[7:474, 7:632]
    cv2.imwrite(str(depth_path), img_depth_uint16)
 
    #save filled depth image
    img_rawDepth = depth_raw * 1000.0
    img_rawDepth_uint16 = img_rawDepth.astype(np.uint16)
    rawDepth_path=(folder / "rawDepth")
    if not os.path.exists(rawDepth_path):
        os.makedirs(rawDepth_path)
    rawDepth_path=rawDepth_path / (str(idx) +".png")
    cv2.imwrite(str(rawDepth_path), img_rawDepth_uint16)

    #save RGB image
    image = image[:, :, ::-1]
    image_black_boundary = np.zeros((480, 640, 3), dtype=np.uint8)
    image_black_boundary[7:474, 7:632, :] = image[7:474, 7:632, :]
    rgb_path=(folder / "rgb")
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)
    rgb_path=rgb_path / (str(idx) +".png")
    #crop away the boundary
    # image=image[7:474, 7:632, :]
    cv2.imwrite(str(rgb_path), image_black_boundary)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage: %s <h5_file> <out_folder>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    h5_file = h5py.File(sys.argv[1], "r")
    # h5py is not able to open that file. but scipy is
    # train_test = scipy.io.loadmat(sys.argv[2])
    out_folder = sys.argv[2]
    #OS independent path
    out_folder=Path(out_folder)

    # test_images = set([int(x) for x in train_test["testNdxs"]])
    # train_images = set([int(x) for x in train_test["trainNdxs"]])
    # print("%d training images" % len(train_images))
    # print("%d test images" % len(test_images))

    filledDepth = h5_file['depths']
    depth_raw = h5_file['rawDepths']

    print("reading", sys.argv[1])

    images = h5_file['images']
    scenes = [u''.join(chr(c[0]) for c in h5_file[obj_ref]) for obj_ref in h5_file['sceneTypes'][0]]

    print("processing images")
    for i, image in enumerate(images):
        print("image", i + 1, "/", len(images))
        convert_image(i, filledDepth[i, :, :].T,depth_raw[i,:,:].T,image.T)

    print("Finished")


#     import matplotlib.pyplot as plt

#     h5_file = h5py.File(r'C:\Users\lahir\data\matlabfiles\nyu_depth_v2_labeled.mat', "r")
#     depth_raw = h5_file['rawDepths']
#     d=depth_raw[10, :, :].T
#     plt.imshow(d)
#     plt.show()

# img=plt.imread('C:\\Users\\lahir\\data\\nyu_depth_v2\\refocused\\train\\rgb\\100.png')
# np.min(img),np.max(img)



