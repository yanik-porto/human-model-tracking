import cv2
from random import seed
from random import randint

seed(1)

def draw_keypoints(image, hpSkels):
    imageOverlay = image.copy()
    for hpSkel in hpSkels:
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        for i, kpt in enumerate(hpSkel.skeleton):
            kptInt = kpt[:2].astype('int32')
            thickness = 3 if hpSkel.confidences is None else int(hpSkel.confidences[i] * 1000)
            if thickness > 10:
                thickness = 10
            cv2.circle(imageOverlay, kptInt, radius=2, color=color, thickness=thickness)
    return imageOverlay