from .HPEstimationHRNET import HPEstimationHRNET
from .HPEstimationAlphaPose import HPEstimationAlphaPose
from .HPEstimationHMR import HPEstimationHMR
from .HPEstimationYoloPose import HPEstimationYoloPose

def Create(config):
    estimator = None

    if config.estimator == "hrnet":
        estimator = HPEstimationHRNET()
    elif config.estimator == "alpha_pose":
        estimator = HPEstimationAlphaPose()
    elif config.estimator == "hmr":
        estimator = HPEstimationHMR()
    elif config.estimator == "yolo_pose":
        estimator = HPEstimationYoloPose()
    else:
        print("Unknown estimator")

    return estimator
