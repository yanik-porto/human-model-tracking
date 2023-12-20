class Config():
    def __init__(self):
        self.estimator = "hrnet"
        self.target_fps = 25
        self.sampling_by_sec = 2
        self.disp = True
        self.verbose = True

def load_config():
    config = Config()
    return config
    