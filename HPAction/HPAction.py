import HP.HP as HP
from .preprocessing import *

class HPAction(object):
    def __init__(self):
        pass

    def process(self, hps):
        pass

    def load_preprocessor(self, config):
        steps = create_preprocessing(config)
        return steps