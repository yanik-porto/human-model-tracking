from .HP import HP

class HPCoco(HP):
    def __init__(self, image, skeleton):
        bbox = self.skeleton_to_bbox(skeleton)
        super(HPCoco, self).__init__(image, bbox)
        self.skeleton = skeleton