from HPEstimation.HPEstimationHMR import HPEstimationHMR
from HPEstimation import HPEstimationFactory 
from HPAction.HPActionUNIK import HPActionUNIK
from HPTracker.HPTracker import HPTracker
from HPTracker.HPTrackerDS import HPTrackerDS
from settings.utils import AverageMeter
from .Visualizer import Visualizer
from .SMPL2VERTS import SMPL2VERTS

import numpy as np
import cv2
import time
import os

class HPManager:
    def __init__(self, config):
        self.config = config

        # 2D Pose estimator 
        self.estimator = HPEstimationFactory.Create(config)
        
        # Shape estimator from 2D
        self.estimator_shape = None
        if config.estimator_shape is not None:
            self.estimator_shape = HPEstimationHMR()

        # action classifier
        self.action_sensor = HPActionUNIK()

        # tracker for each view
        self.trackers = {}

        # utils
        self.estim_time = AverageMeter()
        self.tracking_time = AverageMeter()

        self.smpl2verts = SMPL2VERTS(load_rest=True) if self.config.estimator == "hmr" else None
        self.visualizer = Visualizer(self.config, self.smpl2verts) if self.config.disp else None

    def process(self, himages):
        # 2D pose estimation
        st = time.time()
        hps = self.estimator.process(himages)
        self.estim_time.update(time.time() - st)

        # Shape estimation
        if self.estimator_shape is not None:
            self.estimate_shape(himages, hps)

        # tracking
        st = time.time()
        for himg in himages:
            if himg.viewpointId not in self.trackers:
                self.trackers[himg.viewpointId] = HPTracker(himg.viewpointId, self.config.keep_only_one)
            if himg.viewpointId in hps:
                self.trackers[himg.viewpointId].process(hps[himg.viewpointId])
        self.tracking_time.update(time.time() - st)
        
        # estimate action
        if self.config.estimate_action:
            self.get_action()

        if self.config.disp:
            self.visualizer.visualize_all(himages, hps)

        if self.config.verbose:
            print("time estimation : {est_time.avg:.3f}\t".format(est_time=self.estim_time))

    def estimate_shape(self, himages, hpsByImg):
        for himg in himages:
            hpsImg = hpsByImg[himg.viewpointId]
            hpsToProcess = []
            for hp in hpsImg:
                if len(hp.confidences) > 0:
                    goodJoints = np.sum(np.array(hp.confidences) > self.estimator.minScoreKpt)
                    ratio = goodJoints / len(hp.confidences)
                    print("ratio: ", ratio)
                    if ratio > 0.7:
                        hpsToProcess.append(hp)
            if len(hpsToProcess) > 0:
                self.estimator_shape.process_hps(himg, hpsToProcess)

    def save_projections(self):
        for tracker in self.trackers.values():
            tracker.save_tracklets()

    def get_action(self):
        for viewId in self.trackers:
            for trackid, hps in self.trackers[viewId].tracklets.items():
                if len(hps) > self.action_sensor.maxlen:
                    hps = hps[-self.action_sensor.maxlen:]

                action_pred = self.action_sensor.process(hps)
                hps[-1].lastAction = action_pred