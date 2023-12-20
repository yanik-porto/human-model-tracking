from HPEstimation.HPEstimation import HPEstimation
from HPEstimation import HPEstimationFactory 
from settings.utils import AverageMeter

import time
import cv2
import os
from HP.utils import draw_keypoints

class Pipeline():
    def __init__(self, config):
        self.config = config
        self.estimator = HPEstimationFactory.Create(config)
        self.projector = Projector()
        self.tracklets = {}

        self.estim_time = AverageMeter()

    def set_viewpoints(self, metaByVpath):
        self.metaByVid = {}
        for vpath in metaByVpath:
            self.metaByVid[os.path.basename(vpath)] = metaByVpath[vpath]

    def process(self, himages):
        st = time.time()
        hps = self.estimator.process(himages)
        self.estim_time.update(time.time() - st)

        if self.config.disp:
            for himg in himages:
                hpsImg = [hp for hp in hps if hp.viewpointId == himg.viewpointId]
                imgOverlay = draw_keypoints(himg.data, hpsImg)
                imgOverlay = cv2.cvtColor(imgOverlay, cv2.COLOR_RGB2BGR)
                dispIm = cv2.resize(imgOverlay, (960, 540))
                cv2.imshow("mywind", dispIm)
                cv2.waitKey(1000)
        for hp in hps:
            if hp.viewpointId not in self.tracklets:
                self.tracklets[hp.viewpointId] = []
            self.tracklets[hp.viewpointId].append(hp)
        if self.config.verbose:
            print("time estimation : {est_time.avg:.3f}\t".format(est_time=self.estim_time))