from .HPEstimation import HPEstimation
from HP.HPCoco import HPCoco

import torch
import torch.nn as nn
import torchvision
from yacs.config import CfgNode as CN
import yaml
from easydict import EasyDict as edict
import cv2
import numpy as np
import os

from .AlphaPose.fastpose import FastPose
from .utils import letterbox_image, heatmap_to_coord

def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config

def pre_process(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    print(img.shape)
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

class HPEstimationAlphaPose(HPEstimation):
    def __init__(self):
        super(HPEstimationAlphaPose, self).__init__()

        if not os.path.isfile("HPEstimation/AlphaPose/256x192_res50_lr1e-3_1x.yaml"):
            print("no config file found")
        cfg = update_config("HPEstimation/AlphaPose/256x192_res50_lr1e-3_1x.yaml")
        # print(cfg)
        self.inp_dim = cfg.get('INP_DIM', 608)

        cfg['PRESET'] = cfg['DATA_PRESET']
        self.eval_joints = [*range(0,26)]
        self.norm_type = cfg.LOSS.get('NORM_TYPE', None)
        self.hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self.model = FastPose(norm_layer=nn.BatchNorm2d,**cfg)
        self.model.load_state_dict(torch.load("HPEstimation/AlphaPose/checkpoints/halpe26_fast_res50_256x192.pth"))

    def process(self, himages):
        dets = []
        for himage in himages:
            dets.extend(self.process_image(himage))
        return dets

    
    def process_image(self, himage):
        image = himage.data

        base_size = image.shape

        hm_data = []
        img, _, _ = pre_process(image, self.inp_dim)
        out = self.model(img)
        hm_data = out

        bbox = (0, 0, base_size[0], base_size[1])

        pose_coords_body_foot, pose_scores_body_foot = heatmap_to_coord(
                            hm_data[0][self.eval_joints], bbox, hm_shape=self.hm_size, norm_type=self.norm_type)
        
        dets = []
        dets.append(HPCoco(himage, pose_coords_body_foot, pose_scores_body_foot))

        return dets

