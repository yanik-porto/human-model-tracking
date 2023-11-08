import sys
import os
sys.path.insert(0, os.getcwd())
import unittest
import cv2

from HPEstimation.HPEstimationHRNET import HPEstimationHRNET

class TestHumanPoseEstimation(unittest.TestCase):
    def test_hrnet(self):
        estimator = HPEstimationHRNET()
        imagePath = "tests/000000000785.jpg"
        img = cv2.imread(imagePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = estimator.process([img])

if __name__ == '__main__':
    unittest.main()