from .pose_related import *
from .formatting import *
from .utils import *
from .sampling import *

def create_preprocessing(config):
    assert 'preprocessing' in config, "no preprocessing specified in config"

    cfg_preprocessing = config['preprocessing']

    steps = []

    for step in cfg_preprocessing:
        if step == "ToTensor":
            steps.append(ToTensor(**cfg_preprocessing[step]))
        if step == "Collect":
            steps.append(Collect(**cfg_preprocessing[step]))
        if step == "PreNormalize2D":
            steps.append(PreNormalize2D(**cfg_preprocessing[step]))
        if step == "GenSkeFeat":
            steps.append(GenSkeFeat(**cfg_preprocessing[step]))
        if step == "PoseDecode":
            steps.append(PoseDecode())
        if step == "FormatGCNInput":
            steps.append(FormatGCNInput(**cfg_preprocessing[step]))
        if step == "FormatGCNInputMV":
            steps.append(FormatGCNInputMV(**cfg_preprocessing[step]))
        if step == "UniformSample":
            steps.append(UniformSample(**cfg_preprocessing[step]))
        if step == "Resample":
            steps.append(Resample(**cfg_preprocessing[step]))
        if step == "Coco2H36m":
            steps.append(Coco2H36m())
    return Compose(steps)
