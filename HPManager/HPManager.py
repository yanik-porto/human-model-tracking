from HPEstimation.HPEstimationHMR import HPEstimationHMR
from HPEstimation import HPEstimationFactory 
from HPAction.HPActionSTGCN import HPActionSTGCN
from HPTracker.HPTtracker import HPTracker
from settings.utils import AverageMeter
from HP.utils import draw_keypoints

import numpy as np
import cv2
import time

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
        self.action_sensor = HPActionSTGCN()

        # tracker for each view
        self.trackers = {}

        # utils
        self.estim_time = AverageMeter()


    def process(self, himages):
        # 2D pose estimation
        st = time.time()
        hps = self.estimator.process(himages)
        self.estim_time.update(time.time() - st)

        # Shape estimation
        if self.estimator_shape is not None:
            self.estimate_shape(himages, hps)

        for himg in himages:
            if himg.viewpointId not in self.trackers:
                self.trackers[himg.viewpointId] = HPTracker(himg.viewpointId)
            self.trackers[himg.viewpointId].process(hps[himg.viewpointId])
            
        if self.config.disp:
            for (viewId, hpsImg) in hps.items():
                print("draw ", len(hpsImg), " skeletons")
                himg = next(himg for himg in himages if himg.viewpointId == viewId)
                if not himg:
                    continue

                imgOverlay = draw_keypoints(himg.data, hpsImg)
                imgOverlay = cv2.cvtColor(imgOverlay, cv2.COLOR_RGB2BGR)
                dispIm = cv2.resize(imgOverlay, (960, 540))
                cv2.imshow(viewId, dispIm)
                cv2.waitKey(100)

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

    # if self.config.estimate_action:
    #     for viewId in self.tracklets:
    #         if len(self.tracklets[viewId]) == self.action_sensor.maxlen:
    #             print("Get action from view #", viewId)
    #             self.action_sensor.process(self.tracklets[viewId])
