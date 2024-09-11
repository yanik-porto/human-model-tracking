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

colors = []
for clridx in range(30):
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

def draw_bbox(image, hps):
    imageOverlay = image.copy()
    for hp in hps:
        color = colors[hp.trackid]
        xyxy = hp.xyxy()
        cv2.rectangle(imageOverlay, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]) , color)
    return imageOverlay

def draw_keypoints(image, hpSkels):
    imageOverlay = image.copy()
    for hpSkel in hpSkels:
        if hpSkel.trackid is not None and hpSkel.trackid != -1:
            color = colors[hpSkel.trackid]
        else:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
        for i, kpt in enumerate(hpSkel.skeleton):
            kptInt = kpt[:2].astype('int32')
            thickness = 3 if len(hpSkel.confidences) == 0 else int(hpSkel.confidences[i] * 1000)
            if thickness > 10:
                thickness = 10
            cv2.circle(imageOverlay, kptInt, radius=2, color=color, thickness=thickness)
            cv2.putText(imageOverlay, coco_part_labels[i], kptInt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        # draw bbox
        xyxy = hpSkel.xyxy()
        cv2.rectangle(imageOverlay, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]) , color)

        # draw trackid
        if hpSkel.trackid != -1:
            cv2.putText(imageOverlay, str(hpSkel.trackid), ((xyxy[2] + xyxy[0]) // 2, (xyxy[3] + xyxy[1])//2 ), cv2.FONT_HERSHEY_DUPLEX, 3, color, thickness=3)

    return imageOverlay

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[2], b[2]) - x
  h = min(a[3], b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)

def iou_bbox(bbox1, bbox2):

    inter = intersection(bbox1, bbox2)
    if len(inter) < 4:
        return 0.0

    areaInter = inter[2] * inter[3]

    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    areaUnion = area1 + area2 - areaInter

    iou = areaInter / areaUnion if areaUnion > 0 else 0

    return iou

def nms(hps, nmsth):
    if len(hps) == 0:
        return hps
    
    if hasattr(hps[0], 'confidences') and len(hps[0].confidences) > 0:
        hps = sorted(hps, key=lambda x: sum(x.confidences), reverse=True)

    hpsOut = []
    for hp in hps:
        keep = True
        for hpout in hpsOut:
            iou = iou_bbox(hp.xyxy(), hpout.xyxy())
            keep = iou <= nmsth
            if not keep:
                break
        if keep:
            hpsOut.append(hp)
        
    return hpsOut