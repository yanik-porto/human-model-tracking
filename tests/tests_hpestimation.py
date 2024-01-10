import sys
import os
sys.path.insert(0, os.getcwd())
import unittest
import cv2

from HPEstimation.HPEstimationHRNET import HPEstimationHRNET
from HPEstimation.HPEstimationAlphaPose import HPEstimationAlphaPose
from HP.utils import draw_keypoints
from SensorData.HImage import HImage
class TestHumanPoseEstimation(unittest.TestCase):
    def test_hrnet(self):
        estimator = HPEstimationHRNET()
        imagePath = "tests/000000000785.jpg"
        img = cv2.imread(imagePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        himg = HImage(img, os.path.basename(imagePath))
        dets = estimator.process([himg])
        print(len(dets), " skeleton detected")

        imgOverlay = draw_keypoints(img, dets)
        imgOutPath = imagePath.replace(".jpg", "_kpts_" + estimator.__class__.__name__ + ".jpg")
        imgOverlay = cv2.cvtColor(imgOverlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imgOutPath, imgOverlay)

    def test_alpha_pose(self):
        estimator = HPEstimationAlphaPose()
        imagePath = "tests/000000000785.jpg"
        img = cv2.imread(imagePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        himg = HImage(img, os.path.basename(imagePath))
        dets = estimator.process([himg])
        print(len(dets), " skeleton detected")

        imgOverlay = draw_keypoints(img, dets)
        imgOutPath = imagePath.replace(".jpg", "_kpts_" + estimator.__class__.__name__ + ".jpg")
        imgOverlay = cv2.cvtColor(imgOverlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imgOutPath, imgOverlay)


if __name__ == '__main__':
    unittest.main()