from .HPEstimationHRNET import HPEstimationHRNET

def Create(config):
    estimator = None

    if config.estimator is "hrnet":
        estimator = HPEstimationHRNET()

    return estimator
