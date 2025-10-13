import sys
import os
sys.path.insert(0, os.getcwd())
import unittest
import cv2
from pathlib import Path
import time

from HPEstimation.HPEstimationHRNET import HPEstimationHRNET
from HPEstimation.HPEstimationAlphaPose import HPEstimationAlphaPose
from HPEstimation.HPEstimationYolo import HPEstimationYolo
from HPEstimation.HPEstimationYoloPose import HPEstimationYoloPose
from HPEstimation.HPEstimationHMR import HPEstimationHMR
from HP.utils import draw_keypoints, draw_bbox
from SensorData.HImage import HImage
from HPManager.Visualizer import Visualizer
from HPManager.SMPL2VERTS import SMPL2VERTS

def run_estimator(estimator, imagePath="tests/000000000785.jpg", save_keypoints=False, save_bbox=False, save_mesh=False):
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    himg = HImage(img, Path(imagePath))
    start_time = time.time()
    dets = estimator.process([himg])
    end_time = time.time()
    print(len(dets), " skeleton detected")

    _, ext = os.path.splitext(imagePath)
    if save_keypoints:
        imgOverlay = draw_keypoints(img, dets[himg.viewpointId])
        imgOutPath = imagePath.replace(ext, "_kpts_" + estimator.__class__.__name__ + ext)
        imgOverlay = cv2.cvtColor(imgOverlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imgOutPath, imgOverlay)

    if save_bbox:
        imgOverlay = draw_bbox(img, dets[himg.viewpointId])
        imgOutPath = imagePath.replace(ext, "_bboxes_" + estimator.__class__.__name__ + ext)
        imgOverlay = cv2.cvtColor(imgOverlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imgOutPath, imgOverlay)

    if save_mesh:
        class cfg:
            def __init__(self):
                self.estimator = 'hmr'
                self.verbose = False
        smpl2verts = SMPL2VERTS(load_rest=False) 
        visu = Visualizer(cfg(), smpl2verts)
        imgOverlay = visu.visualize_all([himg], dets)
        imgOutPath = imagePath.replace(ext, "_meshes_" + estimator.__class__.__name__ + ext)
        cv2.imwrite(imgOutPath, imgOverlay)

    inference_time = end_time - start_time
    print(f"{estimator.__class__.__name__} : Inference time: {inference_time:.6f} seconds")

    return dets

class TestHumanPoseEstimation(unittest.TestCase):
    def test_hrnet(self):
        estimator = HPEstimationHRNET(keepOnlyOne=True)
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
        dets = run_estimator(estimator, save_keypoints=True, save_mesh=True)

if __name__ == '__main__':
    unittest.main()