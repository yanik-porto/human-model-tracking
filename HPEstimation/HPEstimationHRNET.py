from .HPEstimation import HPEstimation
from HP.HPCoco import HPCoco
from .HRNET.pose_higher_hrnet import get_pose_net
from .utils import resize_align_multi_scale, get_multi_stage_outputs, aggregate_results, get_multi_scale_size, get_final_preds
from .HRNET.group import HeatmapParser

import torch
import torchvision
from yacs.config import CfgNode as CN

class HPEstimationHRNET(HPEstimation):
    def __init__(self):
        super(HPEstimationHRNET, self).__init__()

        self.minScoreKpt = 0.01

        self.cfg = self.load_config()
        model = get_pose_net(self.cfg, is_train=False)
        model.load_state_dict(torch.load(self.cfg.TEST.MODEL_FILE), strict=True)
        self.model = torch.nn.DataParallel(model).cuda()
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

        self.input_size = self.cfg.DATASET.INPUT_SIZE
        self.scale_factor = self.cfg.TEST.SCALE_FACTOR
        self.parser = HeatmapParser(self.cfg)

  
    def load_config(self):
        _C = CN()
        _C.MODEL = CN()
        _C.MODEL.NAME = 'pose_multi_resolution_net_v16'
        _C.MODEL.INIT_WEIGHTS = True
        _C.MODEL.PRETRAINED = ''
        _C.MODEL.NUM_JOINTS = 17
        _C.MODEL.TAG_PER_JOINT = True
        _C.MODEL.EXTRA = CN(new_allowed=True)
        _C.MODEL.SYNC_BN = False
        _C.DATASET = CN()
        _C.DATASET.INPUT_SIZE = 512
        _C.DATASET.NUM_JOINTS = 17
        _C.DATASET.MAX_NUM_PEOPLE = 30
        _C.DATASET.WITH_CENTER = False
        _C.TEST = CN()
        # size of images for each device
        # _C.TEST.BATCH_SIZE = 32
        _C.TEST.IMAGES_PER_GPU = 32
        # Test Model Epoch
        _C.TEST.FLIP_TEST = False
        _C.TEST.ADJUST = True
        _C.TEST.REFINE = False
        _C.TEST.SCALE_FACTOR = [1]
        # group
        _C.TEST.DETECTION_THRESHOLD = 0.2
        _C.TEST.TAG_THRESHOLD = 1.
        _C.TEST.USE_DETECTION_VAL = True
        _C.TEST.IGNORE_TOO_MUCH = False
        _C.TEST.MODEL_FILE = ''
        _C.TEST.IGNORE_CENTER = True
        _C.TEST.NMS_KERNEL = 3
        _C.TEST.NMS_PADDING = 1
        _C.TEST.PROJECT2IMAGE = False
        _C.TEST.WITH_HEATMAPS = (True,)
        _C.TEST.WITH_AE = (True,)
        _C.TEST.LOG_PROGRESS = False
        _C.TEST.IGNORE_CENTER = True
        _C.LOSS = CN()
        _C.LOSS.WITH_AE_LOSS = (True,)
        _C.LOSS.WITH_HEATMAPS_LOSS = (True,)
        _C.defrost()
        _C.merge_from_file("HPEstimation/HRNET/config.yaml")
        _C.freeze()
        return _C
    
    def process_image(self, himage):
        image = himage.data
        base_size, center, scale = get_multi_scale_size(
            image, self.cfg.DATASET.INPUT_SIZE, 1.0, min(self.cfg.TEST.SCALE_FACTOR)
        )

        final_heatmaps = None
        tags_list = []

        for idx, s in enumerate(sorted(self.scale_factor, reverse=True)):
            inputs = self.pre_process(image, s)
            outputs = self.model(inputs)
            final_heatmaps, tags_list = self.post_process(outputs, s, base_size, final_heatmaps, tags_list)
        
        final_heatmaps = final_heatmaps / float(len(self.cfg.TEST.SCALE_FACTOR))
        tags = torch.cat(tags_list, dim=4)
        grouped, scores = self.parser.parse(
            final_heatmaps.detach(), tags.detach(), self.cfg.TEST.ADJUST, self.cfg.TEST.REFINE
        )
        final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )
        
        dets = []
        for skel in final_results:
            dets.append(HPCoco(himage, skel[:, :2], skel[:, 2]))

        return dets

    def pre_process(self, image, scale):
        image_resized, center, scale = resize_align_multi_scale(
            image, self.input_size, scale, min(self.scale_factor)
        )
        image_resized = self.transforms(image_resized)
        image_resized = image_resized.unsqueeze(0).cuda()
        return image_resized


    def post_process(self, outputs, s, base_size, final_heatmaps, tags_list):

        outputs, heatmaps, tags = get_multi_stage_outputs(
            self.cfg, outputs,
            self.cfg.TEST.PROJECT2IMAGE, base_size
        )

        final_heatmaps, tags_list = aggregate_results(
            self.cfg, s, final_heatmaps, tags_list, heatmaps, tags
        )

        return final_heatmaps, tags_list
        