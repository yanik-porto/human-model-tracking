import torch
import numpy as np
from smplx import SMPL as _SMPL
from smplx.lbs import vertices2joints


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        J_regressor_extra = np.load('HPEstimation/HMR/data/J_regressor_coco.npy')
        self.register_buffer('J_regressor_CONVENTION', torch.tensor(J_regressor_extra, dtype=torch.float32))

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        conv_joints = vertices2joints(self.J_regressor_CONVENTION, smpl_output.vertices)
        return conv_joints, smpl_output.vertices
