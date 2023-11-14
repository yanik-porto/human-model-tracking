class HP:
    def __init__(self, image, bbox):
        self.image = image
        # bbox : left top width height
        self.bbox = bbox
        self.viewpointId = self.image.viewpointId