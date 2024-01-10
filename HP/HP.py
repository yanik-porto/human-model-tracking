class HP:
    def __init__(self, image, bbox):
        self.image = image
        # bbox : left top width height
        self.bbox = bbox
        self.viewpointId = self.image.viewpointId

    def xyxy(self):
        xmin = int(self.bbox[0])
        ymin = int(self.bbox[1])
        xmax = int(self.bbox[0] + self.bbox[2])
        ymax = int(self.bbox[1] + self.bbox[3])
        print(xmin)
        return xmin, ymin, xmax, ymax