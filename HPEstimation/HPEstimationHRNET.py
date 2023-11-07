from .HPEstimation import HPEstimation
from .HRNET.pose_higher_hrnet import get_pose_net
from HRNET.utils import resize_align_multi_scale

import torch
import torchvision
from yacs.config import CfgNode as CN

class HPEstimationHRNET(HPEstimation):
    def __init__(self):
        super(HPEstimationHRNET, self).__init__()

        cfg = self.load_config()
        model = get_pose_net(cfg, is_train=False)
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
        self.model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
        self.model.eval()

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        self.input_size = cfg.DATASET.INPUT_SIZE
        self.scale_factor = cfg.TEST.SCALE_FACTOR
  
    def load_config(self):
        _C = CN()
        _C.defrost()
        _C.merge_from_file("config.yaml")
        _C.freeze()
        return _C

    def process(self, images):
        # assert 1 == images.size(0), 'Test batch size should be 1'
        # image = images[0].cpu().numpy()
        assert 1 == len(images), 'Test batch size should be 1'
        image = images[0]

        for idx, s in enumerate(sorted(self.scale_factor, reverse=True)):
            inputs = self.pre_process(image, s)
            outputs = self.model(inputs)
            outputs = self.post_process(outputs) # voir si pas à la fin

        return dets

    def pre_process(self, image, scale):
        image_resized, center, scale = resize_align_multi_scale(
            image, self.input_size, scale, min(self.scale_factor)
        )
        image_resized = self.transforms(image_resized)
        return image_resized


    def post_process(self, outputs):

        # considering multibatch
        for i, output in enumerate(outputs):
            if len(outputs) > 1 and i != len(outputs) - 1:
                output = torch.nn.functional.interpolate(
                    output,
                    size=(outputs[-1].size(2), outputs[-1].size(3)),
                    mode='bilinear',
                    align_corners=False
                )
        