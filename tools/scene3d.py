import numpy as np
import cv2

class Scene3D:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, img_res_height=None):
        height = img_res if img_res_height is None else img_res_height
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, height // 2]

    def project_joints(self, joints, camera_translation):
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation

        K = np.eye(3)
        K[0][0] = self.focal_length
        K[1][1] = self.focal_length
        K[0][2] = self.camera_center[0]
        K[1][2] = self.camera_center[1]

        Khomo = K @ np.concatenate((np.eye(3),np.zeros((3,1))), axis=1)
        P = Khomo @ camera_pose
        joints2d = P @ np.concatenate((joints, np.ones((joints.shape[0], 1))), axis=1).transpose()
        joints2d[:, :] /= joints2d[2, :]
        return joints2d[:2, :].transpose()


    def render_joints(self, joints, camera_translation, image):
        joints2d = self.project_joints(joints, camera_translation)
        imageOverlay = image.copy()
        for ikpt in range(joints2d.shape[0]):
            kptInt = (int(joints2d[ikpt][0]), int(joints2d[ikpt][1]))
            cv2.circle(imageOverlay, kptInt, radius=2, color=(0,0,255), thickness=3)

        return imageOverlay
