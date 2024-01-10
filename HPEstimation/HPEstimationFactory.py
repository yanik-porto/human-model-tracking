from .HPEstimationHRNET import HPEstimationHRNET
from .HPEstimationAlphaPose import HPEstimationAlphaPose

def Create(config):
    estimator = None

    if config.estimator == "hrnet":
        estimator = HPEstimationHRNET()
    elif config.estimator == "alpha_pose":
        estimator = HPEstimationAlphaPose()
    else:
        print("Unknown estimator")

    return estimator
