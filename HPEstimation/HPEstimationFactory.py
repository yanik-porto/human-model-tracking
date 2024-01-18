from .HPEstimationHRNET import HPEstimationHRNET
from .HPEstimationAlphaPose import HPEstimationAlphaPose
from .HPEstimationHMR import HPEstimationHMR

def Create(config):
    estimator = None

    if config.estimator == "hrnet":
        estimator = HPEstimationHRNET()
    elif config.estimator == "alpha_pose":
        estimator = HPEstimationAlphaPose()
    elif config.estimator == "hmr":
        estimator = HPEstimationHMR()
    else:
        print("Unknown estimator")

    return estimator
