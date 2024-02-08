from .HP import HP
import numpy as np

class HPCoco(HP):
    def __init__(self, image, skeleton, confidences=[], bbox=None, trackid=-1):
        bbox = bbox if bbox is not None else self.skeleton_to_bbox(skeleton, confidences)
        super(HPCoco, self).__init__(image, bbox, trackid=trackid)

        # self.skeleton : list of x, y, score
        self.skeleton = []
        for i in range(skeleton.shape[0]):
            self.skeleton.append(skeleton[i, :2])
        self.confidences = confidences

    def skeleton_to_bbox(self, skeleton, confidences=[]):
        skel_filt = skeleton
        if len(confidences) > 0:
          skel_conf = skeleton[confidences > 0.01, :]
          if len(skel_conf) > 4:
            skel_filt = skel_conf

        left = np.amin(skel_filt[:, 0], axis=0)
        right = np.amax(skel_filt[:, 0], axis=0)
        top = np.amin(skel_filt[:, 1], axis=0)
        bottom = np.amax(skel_filt[:, 1], axis=0)
        return np.array((left, top, right - left, bottom - top))
