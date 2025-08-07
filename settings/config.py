class Config():
    def __init__(self):
        # self.estimator = "hrnet"
        # self.estimator = "yolo_pose"
        self.estimator = "hmr"
        self.estimator_shape = None
        self.target_fps = 10
        self.sampling_by_sec = 2
        self.disp = True
        self.verbose = False
        self.estimate_action = False
        # self.save_projections = True
        self.keep_only_one = False

def load_config():
    config = Config()
    return config
    