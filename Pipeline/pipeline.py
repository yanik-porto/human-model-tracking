from HPEstimation.HPEstimationHMR import HPEstimationHMR
from HPEstimation import HPEstimationFactory 
from Projector import Projector
from settings.utils import AverageMeter
import time
import cv2
import os
import numpy as np
from HP.utils import draw_keypoints
from pathlib import Path

class Pipeline():
    def __init__(self, config):
        self.config = config
        self.estimator = HPEstimationFactory.Create(config)
        
        self.estimator_shape = None
        if config.estimator_shape is not None:
            self.estimator_shape = HPEstimationHMR()
        self.projector = Projector()
        self.tracklets = {}

        self.estim_time = AverageMeter()

        self.hpsByViewid = {}

    def set_viewpoints(self, metaByVpath):
        self.metaByVid = {}
        for vpath in metaByVpath:
            self.metaByVid[Path(vpath).stem] = metaByVpath[vpath]

        self.projector.set_viewpoints(self.metaByVid)

    def process(self, himages):
        st = time.time()
        hps = self.estimator.process(himages)
        self.estim_time.update(time.time() - st)

        for himg in himages:
            if himg.viewpointId not in self.hpsByViewid:
                self.hpsByViewid[himg.viewpointId] = []
            hpsImg = [hp for hp in hps if hp.viewpointId == himg.viewpointId]
            if len(hpsImg) == 0:
                print("Problem for saving, add an empty projection instead")
                continue
            self.hpsByViewid[himg.viewpointId].append(hpsImg[0])


        if self.config.disp:
            for himg in himages:
                hpsImg = [hp for hp in hps if hp.viewpointId == himg.viewpointId]
                imgOverlay = draw_keypoints(himg.data, hpsImg)
                imgOverlay = cv2.cvtColor(imgOverlay, cv2.COLOR_RGB2BGR)
                dispIm = cv2.resize(imgOverlay, (960, 540))
                cv2.imshow(himg.viewpointId, dispIm)
                cv2.waitKey(100)

        if self.estimator_shape is not None:
            self.estimate_shape(himages, hps)

        # we estimate only one person per image
        hpsByPerson = {}
        if len(hps) > 0:
            print("project")
            hpsForP1 = []
            hpsForP1 = hps
            hpsByPerson["P1"] = hpsForP1  
            skel3d = self.projector.process(hpsByPerson, self.estimator.minScoreKpt)
        for hp in hps:
            if hp.viewpointId not in self.tracklets:
                self.tracklets[hp.viewpointId] = []
            self.tracklets[hp.viewpointId].append(hp)
        if self.config.verbose:
            print("time estimation : {est_time.avg:.3f}\t".format(est_time=self.estim_time))

    def estimate_shape(self, himages, hps):
        for himg in himages:
            hpsImg = [hp for hp in hps if hp.viewpointId == himg.viewpointId]
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

    def save_projections(self, outFolder):
        for viewid, hps in self.hpsByViewid.items():
            outPath = os.path.join(outFolder, viewid + '.npz')

            keypoints = []
            keypoints_score = []
            for hp in hps:
                keypoints.append(hp.skeleton)
                keypoints_score.append(hp.confidences)
            
            keypoints = np.asarray(keypoints, dtype=np.float32)
            keypoints_score = np.asarray(keypoints_score, dtype=np.float32)
            print(keypoints.shape)
            print(keypoints_score.shape)

            np.savez(outPath, keypoint=keypoints, keypoint_score=keypoints_score)
