import HM.HM as HM
from HM.HMSkeleton import HMSkeleton

import cv2
import numpy as np

class Projector:
    def __init__(self):
        pass

    def set_viewpoints(self, metaByVid):
        self.metaByVid = metaByVid

    def process(self, hpsByPerson):

        skelWithP = []
        for person in hpsByPerson:
            for hp in hpsByPerson[person]:
                P = self.metaByVid[hp.viewpointId]["P"]
                skel = np.array(hp.skeleton, dtype=np.float32).transpose()
                skelWithP.append((skel, P))

        skel3d = cv2.triangulatePoints(skelWithP[0][1], skelWithP[1][1], skelWithP[0][0], skelWithP[1][0])
        # print(skel3d)

        return skel3d.transpose()
        # hms = [HMSkeleton("P1", skel3d)]
        # return hms