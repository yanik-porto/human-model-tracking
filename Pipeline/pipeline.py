from HPEstimation.HPEstimation import HPEstimation
from HPEstimation import HPEstimationFactory 

class Pipeline():
    def __init__(self, config):
        self.estimator = HPEstimationFactory.Create(config)

    def process(self, images):
        hps = self.estimator.process(images)

        