from .HPEstimationHRNET import HPEstimationHRNET
from .HPEstimationAlphaPose import HPEstimationAlphaPose
from .HPEstimationHMR import HPEstimationHMR
from .HPEstimationYoloPose import HPEstimationYoloPose

def Create(config):
    estimator = None

    if config.estimator == "hrnet":
        estimator = HPEstimationHRNET(config.verbose, config.keep_only_one)
    elif config.estimator == "alpha_pose":
        estimator = HPEstimationAlphaPose(config.verbose, config.keep_only_one)
    elif config.estimator == "hmr":
        estimator = HPEstimationHMR(config.verbose, config.keep_only_one)
    elif config.estimator == "yolo_pose":
        estimator = HPEstimationYoloPose(config.verbose, config.keep_only_one)
    else:
        print("Unknown estimator")

    return estimator
