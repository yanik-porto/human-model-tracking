import unittest
import cv2

from HPEstimation import HPEstimationHRNET

class TestHumanPoseEstimation(unittest.TestCase):
    def test_hrnet(self):
        estimator = HPEstimationHRNET()
        imagePath = "/home/yannick/data/EvalDataForModelgenerator/pfb/Images/18_000054.png"
        img = cv2.imread(imagePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

        estimator.process()