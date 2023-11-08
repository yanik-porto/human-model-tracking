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

        imgOverlay = draw_keypoints(img, dets)
        imgOutPath = imagePath.replace(".jpg","_kpts.jpg")
        imgOverlay = cv2.cvtColor(imgOverlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imgOutPath, imgOverlay)

if __name__ == '__main__':
    unittest.main()