from HPEstimation.HPEstimation import HPEstimation
from HPEstimation import HPEstimationFactory 

import cv2
from HP.utils import draw_keypoints
class Pipeline():
    def __init__(self, config):
        self.config = config
        self.estimator = HPEstimationFactory.Create(config)

    def process(self, himages):
        hps = self.estimator.process(himages)

        if self.config.disp:
            for himg in himages:
                hpsImg = [hp for hp in hps if hp.viewpointId == himg.viewpointId]
                imgOverlay = draw_keypoints(himg.data, hpsImg)
                cv2.imshow("mywind", imgOverlay)
                cv2.waitKey(1000)