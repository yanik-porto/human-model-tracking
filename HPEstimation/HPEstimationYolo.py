from ultralytics import YOLO

from .HPEstimation import HPEstimation
from HP.HP import HP

class HPEstimationYolo(HPEstimation):
    def __init__(self, *args):
        super(HPEstimationYolo, self).__init__(*args)

        self.model_det = YOLO("HPEstimation/YOLO/checkpoints/yolov8l.pt")
        self.score_min = 0.1
        self.model_det.conf = self.score_min

    def process_image(self, himage):

        img = himage.data
        preds = self.model_det.predict(img, verbose=self.verbose)
        assert(len(preds) == 1)

        preds = preds[0].boxes
        boxes = preds.xyxy.cpu().numpy()
        classes = preds.cls.cpu().numpy()
        confs = preds.conf.cpu().numpy()
 
        dets = []
        for cl, box, conf in zip(classes, boxes, confs):
            if int(cl) == 0 and conf >= self.score_min :
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
                dets.append(HP(himage, box))
        return dets