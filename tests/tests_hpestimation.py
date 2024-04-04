import sys
import os
sys.path.insert(0, os.getcwd())
import unittest
import cv2
from pathlib import Path

from HPEstimation.HPEstimationHRNET import HPEstimationHRNET
from HPEstimation.HPEstimationAlphaPose import HPEstimationAlphaPose
from HPEstimation.HPEstimationYolo import HPEstimationYolo
from HPEstimation.HPEstimationYoloPose import HPEstimationYoloPose
from HPEstimation.HPEstimationHMR import HPEstimationHMR
from HP.utils import draw_keypoints, draw_bbox
from SensorData.HImage import HImage

def run_estimator(estimator, imagePath="tests/000000000785.jpg", save_keypoints=False, save_bbox=False):
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    himg = HImage(img, Path(imagePath))
    dets = estimator.process([himg])
    print(len(dets), " skeleton detected")

    if save_keypoints:
        imgOverlay = draw_keypoints(img, dets[himg.viewpointId])
        imgOutPath = imagePath.replace(".jpg", "_kpts_" + estimator.__class__.__name__ + ".jpg")
        imgOverlay = cv2.cvtColor(imgOverlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imgOutPath, imgOverlay)

    if save_bbox:
        imgOverlay = draw_bbox(img, dets[himg.viewpointId])
        imgOutPath = imagePath.replace(".jpg", "_bboxes_" + estimator.__class__.__name__ + ".jpg")
        imgOverlay = cv2.cvtColor(imgOverlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imgOutPath, imgOverlay)

    return dets

class TestHumanPoseEstimation(unittest.TestCase):
    def test_hrnet(self):
        estimator = HPEstimationHRNET()
        dets = run_estimator(estimator, save_keypoints=True)

    def test_alpha_pose(self):
        estimator = HPEstimationAlphaPose()
        dets = run_estimator(estimator, save_keypoints=True)

    def test_yolo(self):
        estimator = HPEstimationYolo()
        dets = run_estimator(estimator, save_bbox=True)

    def test_yolo_pose(self):
        estimator = HPEstimationYoloPose()
        dets = run_estimator(estimator, save_keypoints=True)

    def test_hmr(self):
        estimator = HPEstimationHMR()
        dets = run_estimator(estimator, save_keypoints=True)

if __name__ == '__main__':
    unittest.main()