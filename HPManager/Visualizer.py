import numpy as np
import cv2
import torch

from tools.renderer import Renderer
from HP.utils import draw_keypoints
from HM.SMPL.smpl import SMPL
from HM.SMPL import constants
from preprocessing.utils import crop

class Visualizer:
    def __init__(self, cfg):
        self.config = cfg
        self.renderer = None
        self.up_scale = 1
        if cfg.estimator == 'hmr':
            faces = np.load('HPEstimation/HMR/models/smpl/faces.npy')
            self.renderer = Renderer(focal_length = constants.FOCAL_LENGTH, img_res=constants.IMG_RES * self.up_scale, faces=faces) #, img_res_height=constants.IMG_RES)
            # Load SMPL model
            self.smpl = SMPL('HPEstimation/HMR/models/smpl/SMPL_NEUTRAL.pkl',
                        batch_size=1,
                        create_transl=False).cuda()
            

    def visualize_all(self, himages, hps):
        for himg in himages:
            viewId = himg.viewpointId
            imgOverlay = himg.data
            if viewId in hps:
                hpsImg = hps[viewId]
                if self.config.verbose:
                    print("draw ", len(hpsImg), " skeletons")
                imgOverlay = draw_keypoints(imgOverlay, hpsImg)
                if self.renderer is not None:
                    imgOverlay = self.draw_mesh(imgOverlay, hpsImg)
            imgOverlay = cv2.cvtColor(imgOverlay, cv2.COLOR_RGB2BGR)
            dispIm = cv2.resize(imgOverlay, (960, 540))
            cv2.imshow(viewId, dispIm)

            if False:
                if himg.idx != -1:
                    imgPath = os.path.splitext(himg.srcPath)[0] + "_" + str(himg.idx) + ".jpg"
                    cv2.imwrite(imgPath, dispIm)

            cv2.waitKey(1)
        return imgOverlay

    def draw_mesh(self, image, hpMeshs):


        imageOverlay = image.copy()


        for hpMesh in hpMeshs:
            smpl_params = hpMesh.smpl_params
            betas = smpl_params['betas']
            pose = smpl_params['pose']
            camera = smpl_params['cam']
            joints_3d, vertices_3d = self.smpl(betas=betas, body_pose=pose[:,1:], global_orient=pose[:,0].unsqueeze(1), pose2rot=False)

            # Calculate camera parameters for rendering
            camera_translation = torch.stack([camera[:,1], camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * camera[:,0] +1e-9)],dim=-1)
            camera_translation = camera_translation[0].cpu().numpy()
            pred_joints = joints_3d[0].cpu().numpy()
            vertices_3d = vertices_3d[0].cpu().numpy()
            # img_render = img_render.permute(1,2,0).cpu().numpy()

            # crop the same way it is done in HPEstimationHMR
            center, scale = hpMesh.center_scale()
            img_local = crop(imageOverlay, center, scale, (constants.IMG_RES * self.up_scale, constants.IMG_RES * self.up_scale))
            img_local = img_local.astype(np.float32) / 255

            # adapt cam translation to upscale 
            camera_translation[2] /= self.up_scale

            # add cam translation to vertices location (could be passed to renderer instead)
            vertices_3d += camera_translation

            # renderer
            img_local = self.renderer(vertices_3d, [0, 0, 0], img_local)

            # find crop location inside original image 
            xyxy = hpMesh.xyxy()
            w = xyxy[2]-xyxy[0]
            h = xyxy[3]-xyxy[1]
            maxsize = img_local.shape[0]
            if w > h:
                wInLocal = maxsize
                hInLocal = int(h / w * maxsize)
                border = (maxsize - hInLocal) // 2
                img_local = img_local[border:border+hInLocal, 0:wInLocal]
            elif h > w:
                wInLocal = int(w / h * maxsize)
                hInLocal = maxsize
                border = (maxsize - wInLocal) // 2
                img_local = img_local[0:hInLocal, border:border+wInLocal]
            else:
                img_local = img_local

            # copy crop to original image
            img_local = cv2.resize(img_local, (w, h))
            imageOverlay[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = img_local

        return imageOverlay