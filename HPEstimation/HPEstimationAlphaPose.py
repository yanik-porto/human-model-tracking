from .HPEstimationYolo import HPEstimationYolo
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

class HPEstimationAlphaPose(HPEstimationYolo):
    def __init__(self):
        super(HPEstimationAlphaPose, self).__init__()

        if not os.path.isfile("HPEstimation/AlphaPose/256x192_res50_lr1e-3_1x.yaml"):
            print("no config file found")
        cfg = update_config("HPEstimation/AlphaPose/256x192_res50_lr1e-3_1x.yaml")

        cfg['PRESET'] = cfg['DATA_PRESET']
        self.inp_dim = cfg.get('INP_DIM', 608)
        self.eval_joints = [*range(0,26)]
        self.norm_type = cfg.LOSS.get('NORM_TYPE', None)
        self.hm_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self.model = FastPose(norm_layer=nn.BatchNorm2d,**cfg)
        self.model.load_state_dict(torch.load("HPEstimation/AlphaPose/checkpoints/halpe26_fast_res50_256x192.pth"))
    
    def process_image(self, himage):
        dets = []

        hps = super().process_image(himage)

        for  hp in hps:
            roi = hp.bbox_int()
            crop = himage.data[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            img, _, _ = pre_process(crop, self.inp_dim)
            out = self.model(img)

            pose_coords_body_foot, pose_scores_body_foot = heatmap_to_coord(
                                out[0][self.eval_joints], hp.xyxy())

            
            dets.append(HPCoco(himage, pose_coords_body_foot, confidences=pose_scores_body_foot, bbox=hp.bbox))

        return dets

