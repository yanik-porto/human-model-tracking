import argparse
import cv2
from pathlib import Path

from Pipeline.pipeline import Pipeline
from settings.config import load_config
from SensorData.HImage import HImage
from HP.utils import draw_keypoints

def parse_args():
    parser = argparse.ArgumentParser("Read live stream and apply human tracking")
    parser.add_argument("url", type=str, help="Url of the live stream to process (ex: rtsp://192.168.1.88:554/live.sdp)")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    config = load_config()
    config.disp = True
    step = config.target_fps // config.sampling_by_sec

    pipeline = Pipeline(config)
    metainfos = {args.url: {"maxIdx": 60 * 25} } # 1 minute
    pipeline.set_viewpoints(metainfos)

    cap = cv2.VideoCapture(args.url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    idxframe = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if idxframe % step != 0:
            idxframe+=1
            continue

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        himg = HImage(frame, Path("/home/user/dummy"), idxframe)

        images = []
        images.append(himg)
        pipeline.process(images)

        idxframe+=1
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()