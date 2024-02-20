import imageio
import imageio.v3 as iio
import argparse
import os
import sys
import cv2
import numpy as np
import pickle
from pathlib import Path

from settings.config import load_config
from SensorData.HImage import HImage
from Pipeline.pipeline import Pipeline
from settings.utils import load_file_to_matrix

def parse_args():
    parser = argparse.ArgumentParser("Read multiple videos and track people inside")
    parser.add_argument("videos_folder_path", type=str, help="Path to the folder where the videos are stored")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    config = load_config()

    pipeline = Pipeline(config)

    # Collect videos info
    metaByVpath = {}
    for root, _, files in os.walk(args.videos_folder_path):
        for f in files:
            if not f.endswith(('.mkv', '.mp4', '.avi')):
                continue

            fpath = os.path.join(root, f)
            vid = imageio.get_reader(fpath, "ffmpeg")
            meta_data = vid.get_meta_data()
            print(meta_data)

            fpathnoext, _ = os.path.splitext(fpath)
            campath = fpathnoext + '.txt'
            camP = load_file_to_matrix(campath)

            meta = {}
            meta['maxIdx'] = meta_data['duration'] * meta_data['fps']
            meta['P'] = camP

            print(meta)
            metaByVpath[fpath] = meta

    pipeline.set_viewpoints(metaByVpath)

    idx = 0
    while(True):
        print(idx)
        images = []
        for vidpath in metaByVpath:
            if idx < metaByVpath[vidpath]['maxIdx']:
                try:
                    img = iio.imread(vidpath, index=idx)
                    images.append(HImage(img, Path(vidpath).stem))
                except:
                    print("failed reading image from ", vidpath, " at index #", idx)
                    continue
        if len(images) == 0:
            break

        pipeline.process(images)

        # cv2.imshow("mywind", images[0])
        # cv2.waitKey(1000 // config.target_fps // config.sampling_by_sec)
        idx += config.target_fps // config.sampling_by_sec
