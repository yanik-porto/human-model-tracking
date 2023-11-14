from abc import ABC, abstractmethod

class HPAssication(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, hps):
        pass