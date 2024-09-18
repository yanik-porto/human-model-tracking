from .HPEstimationHRNET import HPEstimationHRNET
from .HPEstimationAlphaPose import HPEstimationAlphaPose
from .HPEstimationHMR import HPEstimationHMR
from .HPEstimationYoloPose import HPEstimationYoloPose

def Create(config):
    estimator = None

    if config.estimator == "hrnet":
        estimator = HPEstimationHRNET(config.verbose)
    elif config.estimator == "alpha_pose":
        estimator = HPEstimationAlphaPose(config.verbose)
    elif config.estimator == "hmr":
        estimator = HPEstimationHMR(config.verbose)
    elif config.estimator == "yolo_pose":
        estimator = HPEstimationYoloPose(config.verbose)
    else:
        print("Unknown estimator")

    return estimator
