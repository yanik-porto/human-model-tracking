class HImage():
    def __init__(self, data, srcPath):
        self.data = data
        self.srcPath = srcPath
        self.viewpointId = srcPath.stem