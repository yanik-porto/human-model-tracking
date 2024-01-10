import cv2
from random import seed
from random import randint

seed(1)

coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r', # end of normal coco label
    'head', 'neck', 'hip', 'btoe_l', 'btoe_r', 'stoe_l', 'stoe_r', 'heel_l', 'heel_r'
]

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
            cv2.putText(imageOverlay, coco_part_labels[i], kptInt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    return imageOverlay