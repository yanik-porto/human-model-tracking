import HM.HM as HM

from abc import ABC, abstractmethod

class HPEstimation(ABC):
    def __init__(self):
        self.minScoreKpt = 0.1

    def process(self, himages):
        dets = []
        for himage in himages:
            dets.extend(self.process_image(himage))
        return dets
    
    @abstractmethod
    def process_image(self, himage):
        pass