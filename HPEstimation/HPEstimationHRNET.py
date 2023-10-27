from .HPEstimation import HPEstimation
from .HRNET.pose_higher_hrnet import get_pose_net

import torch
from yacs.config import CfgNode as CN

class HPEstimationHRNET(HPEstimation):
    def __init__(self):
        super(HPEstimationHRNET, self).__init__()

        cfg = self.load_config()
        model = get_pose_net(cfg, is_train=False)
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
        self.model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
        self.model.eval()
  
    def load_config(self):
        _C = CN()
        _C.defrost()
        _C.merge_from_file("config.yaml")
        _C.freeze()
        return _C

    def process(self, images):
        outputs = self.model(images)
        dets = self.post_process(outputs, images)
        return dets

    def post_process(self, outputs, image):
        # considering multibatch
        for i, output in enumerate(outputs):
            if len(outputs) > 1 and i != len(outputs) - 1:
                output = torch.nn.functional.interpolate(
                    output,
                    size=(outputs[-1].size(2), outputs[-1].size(3)),
                    mode='bilinear',
                    align_corners=False
                )
        