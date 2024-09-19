import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
import unittest
import pickle
from pathlib import Path

from HPAction.HPActionUNIK import HPActionUNIK
from HP.HPCoco import HPCoco
from SensorData.HImage import HImage

class TestAction(unittest.TestCase):
    def test_unik(self):
        action_sensor = HPActionUNIK()

        data_path = "tests/babel_mv_sample.pkl"
        with open(data_path, 'rb') as d: data = pickle.load(d)
        sample1 = data
        label = action_sensor.label_map[sample1['label']]
        keypoints = sample1['keypoint'].squeeze()
        kpts_scores = sample1['keypoint_score'].squeeze()

        hps = []
        for i in range(keypoints.shape[0]):
            dummyImg = HImage(np.zeros((10, 10), dtype=int), Path("/home/user/dummy"), i)
            hp = HPCoco(dummyImg, keypoints[i], kpts_scores[i])
            hps.append(hp)

        pred = action_sensor.process(hps)
        
        print(pred)
        assert(pred == label)

if __name__ == '__main__':
    unittest.main()