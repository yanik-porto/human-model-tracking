from .HPCoco import HPCoco

import cv2

class HPSMPL(HPCoco):
    # def __init__(self, image, skeleton, vertices, camera_translation, bbox=None):
    def __init__(self, image, skeleton, img_rendered, bbox=None, trackid=-1):
        super(HPSMPL, self).__init__(image, skeleton, bbox=bbox, trackid=trackid)

        self.img_rendered = img_rendered

        cv2.imshow("smpl", img_rendered)
        cv2.waitKey(1000)
        # self.vertices = vertices
        # self.camera_translation = camera_translation