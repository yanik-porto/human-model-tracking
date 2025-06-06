from ultralytics import YOLO
import os

from .HPEstimation import HPEstimation
from HP.HPCoco import HPCoco, HP

class HPEstimationYoloPose(HPEstimation):
    def __init__(self, *args):
        super(HPEstimationYoloPose, self).__init__(*args)

        self.minScoreKpt = 0.5

        self.model_det = YOLO("HPEstimation/YOLO/checkpoints/yolov8n-pose.pt")
        self.score_min = 0.5
        self.model_det.conf = self.score_min

    def process_image(self, himage):

        dets = []

        img = himage.data
        preds = self.model_det.track(img, persist=True, tracker="bytetrack.yaml", verbose=self.verbose)
        # preds = self.model_det.predict(img)
        assert(len(preds) == 1)

        predsSkel = preds[0].keypoints
        predsBox = preds[0].boxes

        if predsSkel.conf is None:
            return dets

        skels = predsSkel.xy.cpu().numpy()
        confsKpts = predsSkel.conf.cpu().numpy()

        boxes = predsBox.xyxy.cpu().numpy()
        ids = predsBox.id
        confs = predsBox.conf.cpu().numpy()
 
        for idx, (box, conf, kpts, confKpts) in enumerate(zip(boxes, confs, skels, confsKpts)):
            if conf >= self.score_min:
                trackid = -1
                if ids is not None and len(ids) == len(boxes):
                    idBox = ids.cpu().numpy()[idx]
                    trackid = int(idBox)
                    print("idBox: ", idBox)
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
                dets.append(HPCoco(himage, kpts, confKpts, box, trackid=trackid, detscore=conf))
        return dets