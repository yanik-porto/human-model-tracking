from HP.HPCoco import HPCoco
from HP.utils import iou_bbox
from .IdsManager import IdsManager
from SensorData.HImage import HImage

import numpy as np
import os

class HPTracker:
    def __init__(self, trackerid="", keep_only_one=False):
        self.trackerid = trackerid
        self.tracklets = {}
        self.ids_manager = IdsManager(100)
        self.iouMin = 0.1

        # add an empty detection if not found in an image
        self.emptyIfNotFound = True
        self.retrieveNotFoundLaterOn = True
        self.keep_only_one = keep_only_one

    def process(self, hps):
        for hp in hps:
            if hp.trackid == -1:
                if self.keep_only_one and len(self.tracklets) > 0:
                    matchid = list(self.tracklets.keys())[0]
                else:
                    matchid = self.match_prev(hp)
                    if matchid == -1:
                        matchid = self.ids_manager.get_new_id()
                        self.tracklets[matchid] = []
                hp.trackid = matchid
            if not hp.trackid in self.tracklets:
                self.tracklets[hp.trackid] = []
            self.tracklets[hp.trackid].append(hp)

        # fill tracklets with null skeleton if not found in the image
        if self.emptyIfNotFound:
            hpids = [hp.trackid for hp in hps]
            for trackid in self.tracklets.keys():
                if trackid not in hpids:
                    last = self.tracklets[trackid][-1]
                    skelshape = np.asarray(last.skeleton).shape
                    confshape = np.asarray(last.confidences).shape
                    # dummyImg = HImage(None, self.trackerid)
                    dummyImg = last.image # take previous one for getting all path info
                    bboxNotFound =  last.bbox if self.retrieveNotFoundLaterOn else (0, 0, 0, 0)
                    emptyskel = HPCoco(dummyImg, np.zeros(skelshape, np.float32), np.zeros(confshape, np.float32), bbox=bboxNotFound, trackid=trackid)
                    self.tracklets[trackid].append(emptyskel)

    def match_prev(self, hp):
        maxiou = 0.0
        idOfMax = -1
        for (trackid, tracks) in self.tracklets.items():
            last = tracks[-1]
            iou = iou_bbox(hp.xyxy(), last.xyxy())
            if iou > self.iouMin and iou > maxiou:
                maxiou = iou
                idOfMax = trackid

        return idOfMax
    
    def save_tracklets(self):
        for trackletid, hps in self.tracklets.items():
            outFolder = hps[0].image.srcPath.parents[0]
            outPath = os.path.join(outFolder, self.trackerid + '_' + str(trackletid) + '.npz')

            keypoints = []
            keypoints_score = []
            skelshape = np.asarray(hps[0].skeleton).shape
            confshape = np.asarray(hps[0].confidences).shape
            idx = 0
            for hp in hps:
                while idx < hp.image.idx:
                    # add empty skeleton for missing frames
                    keypoints.append(np.zeros(skelshape, np.float32))
                    keypoints_score.append(np.zeros(confshape, np.float32))
                    idx += 1
                keypoints.append(hp.skeleton)
                keypoints_score.append(hp.confidences)
                idx += 1
            
            keypoints = np.asarray(keypoints, dtype=np.float32)
            keypoints_score = np.asarray(keypoints_score, dtype=np.float32)
            print(keypoints.shape)
            print(keypoints_score.shape)

            np.savez(outPath, keypoint=keypoints, keypoint_score=keypoints_score)