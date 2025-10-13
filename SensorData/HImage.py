class HImage():
    def __init__(self, data, srcPath, idx=-1, bbox=None):
        self.data = data
        self.srcPath = srcPath
        self.viewpointId = srcPath.stem
        self.idx = idx
        self.bbox = bbox