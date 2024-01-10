from ultralytics import YOLO

from .HPEstimation import HPEstimation
from HP.HP import HP

class HPEstimationYolo(HPEstimation):
    def __init__(self):
        super(HPEstimationYolo, self).__init__()

        self.model = YOLO("HPEstimation/YOLO/checkpoints/yolov8n.pt")

    def process_image(self, himage):

        img = himage.data
        preds = self.model.predict(img)
        assert(len(preds) == 1)

        preds = preds[0].boxes
        boxes = preds.xyxy.cpu().numpy()
        classes = preds.cls.cpu().numpy()
 
        dets = []
        for cl, box in zip(classes, boxes):
            if int(cl) == 0:
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
                dets.append(HP(himage, box))
        return dets