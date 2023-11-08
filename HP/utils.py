import cv2
from random import seed
from random import randint

seed(1)

def draw_keypoints(image, hpSkels):
    imageOverlay = image.copy()
    for hpSkel in hpSkels:
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        for kpt in hpSkel.skeleton:
            kptInt = kpt[:2].astype('int32')
            cv2.circle(imageOverlay, kptInt, radius=2, color=color, thickness=3)
    return imageOverlay