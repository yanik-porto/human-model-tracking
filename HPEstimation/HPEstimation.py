import HM.HM as HM
from HP.utils import nms

from abc import ABC, abstractmethod

class HPEstimation(ABC):
    def __init__(self):
        self.minScoreKpt = 0.1
        self.nmsth = 0.5

    def process(self, himages):
        dets = {}
        for himage in himages:
            hps = self.process_image(himage)
            hps = nms(hps, self.nmsth)
            dets[himage.viewpointId] = hps
        return dets
    
    @abstractmethod
    def process_image(self, himage):
        pass