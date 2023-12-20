from HPEstimation.HPEstimation import HPEstimation
from HPEstimation import HPEstimationFactory 
from settings.utils import AverageMeter

import time
import cv2
from HP.utils import draw_keypoints

class Pipeline():
    def __init__(self, config):
        self.config = config
        self.estimator = HPEstimationFactory.Create(config)

    def process(self, himages):
        st = time.time()
        hps = self.estimator.process(himages)
        self.estim_time.update(time.time() - st)

        if self.config.disp:
            for himg in himages:
                hpsImg = [hp for hp in hps if hp.viewpointId == himg.viewpointId]
                imgOverlay = draw_keypoints(himg.data, hpsImg)
                cv2.imshow("mywind", imgOverlay)
                cv2.waitKey(1000)
        if self.config.verbose:
            print("time estimation : {est_time.avg:.3f}\t".format(est_time=self.estim_time))