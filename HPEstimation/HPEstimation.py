import HM.HM as HM
from HP.utils import nms

import numpy as np
from abc import ABC, abstractmethod
class HPEstimation(ABC):
    def __init__(self, verbose=False):
        self.minScoreKpt = 0.1
        self.nmsth = 0.5
        self.keepOnlyOne = False
        self.verbose = verbose

    def process(self, himages):
        dets = {}
        for himage in himages:
            hps = self.process_image(himage)
            hps = nms(hps, self.nmsth)
            if self.keepOnlyOne and len(hps) > 1:
                hps = self.keep_only_one(hps)
            dets[himage.viewpointId] = hps
        return dets
    
    @abstractmethod
    def process_image(self, himage):
        pass

    def keep_only_one(self, hps):
        maxConf = 0
        idxMax = -1
        for i, hp in enumerate(hps):
            if len(hp.confidences) == 0:
                idxMax = 0
                break
            conf = np.sum(hp.confidences)
            if conf > maxConf:
                maxConf = conf
                idxMax = i
        if idxMax == -1:
            print("error, idxmax is still -1 : ", len(hps), " estimations found")
            return hps
        
        return [hps[idxMax]]