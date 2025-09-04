from HM.SMPL.smpl import SMPL
from HM.SMPL import constants

import torch
import numpy as np
from tools.rendering.tools.matrix import rotation_3d_x

class SMPL2VERTS():
    def __init__(self, load_rest=False):
        self.smpl = SMPL('HPEstimation/HMR/models/smpl/SMPL_NEUTRAL.pkl',
                    batch_size=1,
                    create_transl=False).cuda()
        
        if load_rest:
            mesh_obj_p = "HPManager/smpl_rest_pose.npz"
            mesh_obj = dict(np.load(mesh_obj_p))
            betas_rest = torch.from_numpy(mesh_obj['betas']).float().cuda()
            poses_rest = torch.from_numpy(mesh_obj['poses']).float().cuda()
            global_orient = poses_rest[:, :3]
            body_pose = poses_rest[:, 3:24*3].reshape(-1, 23*3)
            _, rest_vertices = self.smpl(betas=betas_rest, body_pose=body_pose, global_orient=global_orient, pose2rot=True)
            self.rest_vertices = rest_vertices[0].cpu().numpy()

    def apply_translation(self, vertices_3d, camera, up_scale=1, apply_camera_params=False):
            # Calculate camera parameters for rendering
            camera_translation = torch.stack([-camera[:,1], camera[:,2], 1/camera[:,0]] ,dim=-1)
            if apply_camera_params:
                camera_translation = torch.stack([camera[:,1], camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * camera[:,0] +1e-9)],dim=-1)
            camera_translation = camera_translation[0].cpu().numpy()
            camera_translation[2] /= up_scale  # adapt cam translation to upscale 
            vertices_3d += camera_translation # add cam translation to vertices location (could be passed to renderer instead)

            return vertices_3d

    def __call__(self, hpMesh, up_scale=1, apply_camera_params=True, rotate_to_right_hand=False):
            smpl_params = hpMesh.smpl_params
            betas = smpl_params['betas']
            pose = smpl_params['pose']
            camera = smpl_params['cam']
            joints_3d, vertices_3d = self.smpl(betas=betas, body_pose=pose[:,1:], global_orient=pose[:,0].unsqueeze(1), pose2rot=False)

            pred_joints = joints_3d[0].cpu().numpy()
            vertices_3d = vertices_3d[0].cpu().numpy()

            if rotate_to_right_hand:
                rot = rotation_3d_x(np.radians(180))
                jHomo = np.concatenate((vertices_3d, np.ones((vertices_3d.shape[0], 1))), axis=1).transpose()
                jRotated = rot @ jHomo
                vertices_3d = jRotated.transpose()[:, :3]

            return vertices_3d, camera

            return self.apply_translation(vertices_3d, camera, up_scale, apply_camera_params)
    
    def get_rest_with_trans(self, hpMesh, up_scale=1, apply_camera_params=False):
            smpl_params = hpMesh.smpl_params
            camera = smpl_params['cam']

            camera[:,2] = 1 # all meshs on same plane
            return self.apply_translation(np.copy(self.rest_vertices), camera, up_scale, apply_camera_params)


