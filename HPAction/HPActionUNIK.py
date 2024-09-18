from .HPAction import HPAction
from .UNIK.unik import UNIK
from .Heads.head_single import HeadSingle
import torch
import torch.nn as nn
import numpy as np
import yaml
import os.path as op

class HPActionUNIK(HPAction):
    def __init__(self):
        super(HPActionUNIK, self).__init__()

        params_path = op.join(op.dirname(__file__),"UNIK/params.yaml") 
        cfg = yaml.safe_load(open(params_path))
        cfg_encoder = cfg["encoder"]
        cfg_head = cfg["head"]
        self.encoder = UNIK(**cfg_encoder["params"])
        self.head = HeadSingle(**cfg_head["params"])

        self.encoder.eval()
        self.head.eval()

        chkpt_path = op.join(op.dirname(__file__),"UNIK/unik_head_single.pth") 
        checkpoint = torch.load(chkpt_path)
        self.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        self.head.load_state_dict(checkpoint["head"], strict=True)

        self.preprocessing = self.load_preprocessor(cfg)

        self.label_map = [x.strip() for x in open(op.join(op.dirname(__file__), "babel.txt")).readlines()]

        self.maxlen = cfg["preprocessing"]["UniformSample"]["clip_len"]

    def process(self, hps):
        input_buffer = {}

        keypoints_buffer = []
        kptsscores_buffer = []
        for hp in hps:
            keypoints_buffer.append(hp.skeleton)
            kptsscores_buffer.append(hp.confidences)

        input_buffer["total_frames"] = len(keypoints_buffer)

        keypoints_buffer = np.asarray(keypoints_buffer, dtype=np.float32)
        kptsscores_buffer = np.asarray(kptsscores_buffer, dtype=np.float32)
        keypoints_buffer = np.expand_dims(keypoints_buffer, axis=0)
        kptsscores_buffer = np.expand_dims(kptsscores_buffer, axis=0)
        input_buffer["keypoint"] = keypoints_buffer
        input_buffer["keypoint_score"] = kptsscores_buffer

        input_buffer["img_shape"] = hps[0].image.data.shape[:2]

        prep_buffer = self.preprocessing(input_buffer)

        batch = torch.unsqueeze(prep_buffer['keypoint'], 0)

        feats = self.encoder(batch)
        output = self.head(feats)
        _, pred = output.topk(1, 1, True, True)
        pred = int(pred)

        action = "unknown"
        if pred >= len(self.label_map):
            print("Error: predicted class is not in the label map")

        return self.label_map[pred]