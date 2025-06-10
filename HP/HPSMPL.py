from .HPCoco import HPCoco

import cv2

class HPSMPL(HPCoco):
    # def __init__(self, image, skeleton, vertices, camera_translation, bbox=None):
    def __init__(self, image, skeleton, img_rendered=None, bbox=None, trackid=-1, detscore=-1, smpl_params=None):
        super(HPSMPL, self).__init__(image, skeleton, bbox=bbox, trackid=trackid, detscore=detscore)

        self.img_rendered = img_rendered
        self.smpl_params = smpl_params