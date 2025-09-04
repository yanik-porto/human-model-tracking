import argparse
import cv2
from pathlib import Path
import asyncio
import websockets
import json

from Pipeline.pipeline import Pipeline
from settings.config import load_config
from SensorData.HImage import HImage

def parse_args():
    parser = argparse.ArgumentParser("Read live stream and apply human tracking")
    parser.add_argument("url", type=str, help="Url of the live stream to process (ex: rtsp://192.168.1.88:554/live.sdp)")
    parser.add_argument('--create_web_scene', action='store_true', default=False, help="Send pose estimation via websocket")
    return parser.parse_args()

def process_next_frame(cap, idxframe, step):
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return -1

    if idxframe % step != 0:
        idxframe+=1
        return idxframe

    frame = cv2.resize(frame, (480, 240))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    himg = HImage(frame, Path("/home/user/Demo"), idxframe)

    images = []
    images.append(himg)
    pipeline.process(images)

    idxframe+=1

    return idxframe

if __name__ == '__main__':
    args = parse_args()

    config = load_config()
    config.disp = True
    step = config.target_fps // config.sampling_by_sec

    pipeline = Pipeline(config)
    metainfos = {args.url: {"maxIdx": 60 * 25} } # 1 minute
    pipeline.set_viewpoints(metainfos)

    url = args.url if not args.url.isdigit() else int(args.url)

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    idxframe = 0

    if not args.create_web_scene:
        while idxframe >= 0 :
            idxframe = process_next_frame(cap, idxframe, step)
    else:
        if pipeline.hp_manager.smpl2verts is None:
            print("No smpl loaded")
            exit(1)

        async def handler(websocket, path):
            print("Client connecté")
            await websocket.send(json.dumps({"faces": pipeline.hp_manager.smpl2verts.smpl.faces.tolist()}))  # faces envoyées une seule fois

            idxframe = 0
            while idxframe >= 0:
                idxframe = process_next_frame(cap, idxframe, step)

                verts, trackids = pipeline.last_hms_verts(rest_pose=False)

                if len(verts) < 1:
                    continue

                await websocket.send(json.dumps({"vertices": verts, "trackids":trackids}))


        start_server = websockets.serve(handler, "localhost", 8765)
        print("Serveur WebSocket lancé sur ws://localhost:8765")

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()