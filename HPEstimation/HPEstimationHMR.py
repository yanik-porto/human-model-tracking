from .HPEstimationYolo import HPEstimationYolo
from HM.SMPL.hmr import hmr
from HM.SMPL.smpl import SMPL
from HM.SMPL import constants
from HP.HPCoco import HPCoco
from HP.HPSMPL import HPSMPL
from preprocessing.utils import crop
from tools.renderer import Renderer
from HPEstimation.utils import coords_to_bbox

import numpy as np
import torch
from torchvision.transforms import Normalize
import cv2

class HPEstimationHMR(HPEstimationYolo):
    def __init__(self, *args):
        super(HPEstimationHMR, self).__init__(*args)

        self.model = hmr('HPEstimation/HMR/data/smpl_mean_params.npz').cuda()
        checkpoint = torch.load('HPEstimation/HMR/data/spin_checkpoint.pt', map_location="cuda")
        self.model.load_state_dict(checkpoint['model'], strict=False)

        # Load SMPL model
        self.smpl = SMPL('HPEstimation/HMR/models/smpl/SMPL_NEUTRAL.pkl',
                    batch_size=1,
                    create_transl=False).cuda()
        self.model.eval()

        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        self.renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=self.smpl.faces)

    def pre_process(self, img, hp, input_res):
        center, scale = hp.center_scale()

        img = crop(img, center, scale, (input_res, input_res))
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(img).permute(2,0,1)
        norm_img = self.normalize_img(img.clone())[None]
        return norm_img, img # second image for rendering

    def process_image(self, himage):

        hps = super().process_image(himage)

        return self.process_hps(himage, hps)
    
    def process_hps(self, himage, hps):
        dets = []

        img = himage.data

        for i, hp in enumerate(hps):
            norm_img, img_render = self.pre_process(img, hp, constants.IMG_RES)

            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = self.model(norm_img.cuda())
                pred_joints, pred_vertices = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)

            # Render parametric shape
            # Calculate camera parameters for rendering
            camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
            camera_translation = camera_translation[0].cpu().numpy()
            pred_joints = pred_joints[0].cpu().numpy()
            pred_vertices = pred_vertices[0].cpu().numpy()
            img_render = img_render.permute(1,2,0).cpu().numpy()
            
            if True:
                # Render non-parametric shape
                img_shape = self.renderer(pred_vertices, camera_translation, img_render, pred_joints)
 
            if False:
                # Render side views
                aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
                center = pred_vertices.mean(axis=0)
                rot_vertices = np.dot((pred_vertices - center), aroundy) + center
                img_shape_side = self.renderer(rot_vertices, camera_translation, np.ones_like(img_render), pred_joints)
                # Save reconstructions
                outfile = "output"
                cv2.imwrite(outfile + '_shape' + str(i) +'.png', 255 * img_shape[:,:,::-1])
                cv2.imwrite(outfile + '_shape_side' + str(i) +'.png', 255 * img_shape_side[:,:,::-1])

            joints2d = self.renderer.project_joints(pred_joints, camera_translation)

            joints2d = coords_to_bbox(joints2d, hp.xyxy(), constants.IMG_RES, constants.IMG_RES)
            # dets.append(HPCoco(himage, joints2d, bbox=hp.bbox))
            dets.append(HPSMPL(himage, joints2d, bbox=hp.bbox, img_rendered=img_shape))

        return dets