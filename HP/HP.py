action_default = "unknown"

class HP:
    def __init__(self, image, bbox, trackid=-1, detscore=-1.):
        self.image = image
        # bbox : left top width height
        self.bbox = bbox
        self.viewpointId = self.image.viewpointId
        self.trackid = trackid
        self.lastAction = action_default
        self.detscore = detscore

    def xyxy(self):
        xmin = int(self.bbox[0])
        ymin = int(self.bbox[1])
        xmax = int(self.bbox[0] + self.bbox[2])
        ymax = int(self.bbox[1] + self.bbox[3])
        return xmin, ymin, xmax, ymax
    
    def bbox_int(self):
        return [int(c) for c in self.bbox]
    
    def center_scale(self):
        ul_corner = self.bbox[:2]
        center = ul_corner + 0.5 * self.bbox[2:]
        width = max(self.bbox[2], self.bbox[3])
        scale = width / 200.0
        # make sure the bounding box is rectangular
        return center, scale