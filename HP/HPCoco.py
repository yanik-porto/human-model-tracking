from .HP import HP
import numpy as np

class HPCoco(HP):
    def __init__(self, image, skeleton, confidences=[], bbox=None):
        bbox = bbox if bbox is not None else self.skeleton_to_bbox(skeleton)
        super(HPCoco, self).__init__(image, bbox)

        # self.skeleton : list of x, y, score
        self.skeleton = []
        for i in range(skeleton.shape[0]):
            self.skeleton.append(skeleton[i, :2])
        self.confidences = confidences

    def skeleton_to_bbox(self, skeleton):
        left = np.amin(skeleton[:, 0], axis=0)
        right = np.amax(skeleton[:, 0], axis=0)
        top = np.amin(skeleton[:, 1], axis=0)
        bottom = np.amax(skeleton[:, 1], axis=0)
        return (left, top, right - left, bottom - top)
