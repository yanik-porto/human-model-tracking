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

        skels = []
        Ps = []
        skippedIdx = set()
        for person in hpsByPerson:
            for hp in hpsByPerson[person]:
                if not "P" in self.metaByVid[hp.viewpointId] or self.metaByVid[hp.viewpointId]["P"] is None:
                    print("No projection matrix for viewpoint ", hp.viewpointId, " => abort projection")
                    return None

                P = self.metaByVid[hp.viewpointId]["P"]
                skel = np.array(hp.skeleton, dtype=np.float32)
                for c in range(len(hp.confidences)):
                    conf = hp.confidences[c]
                    if conf < 0.01:
                        skippedIdx.add(c)
                skel = skel.transpose()
                skels.append(skel)
                Ps.append(P)


        for idx in reversed(list(skippedIdx)):
            for s in range(len(skels)):
                skels[s] = np.delete(skels[s], idx, 1)
        skel3d = cv2.triangulatePoints(Ps[0], Ps[1], skels[0], skels[1])

        for idx in skippedIdx:
            skel3d = np.insert(skel3d, idx, [0, 0, 0, 1], axis=1)

        return skel3d.transpose()
        # hms = [HMSkeleton("P1", skel3d)]
        # return hms