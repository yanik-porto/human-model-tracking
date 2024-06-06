from HPManager.HPManager import HPManager
from Projector import Projector
import time
import cv2
import os
import numpy as np
from pathlib import Path

from SensorData.HImage import HImage
from SensorData.utils import get_video_info
from settings.utils import AverageMeter
import imageio.v3 as iio
from tqdm import tqdm

class Pipeline():
    def __init__(self, config):
        self.config = config

        # viewpoint management
        self.hp_manager = HPManager(config)

        # world management
        self.projector = Projector()
        self.skels3d = []

    def set_viewpoints(self, metaByVpath):
        self.metaByVid = {}
        for vpath in metaByVpath:
            self.metaByVid[Path(vpath).stem] = metaByVpath[vpath]

        self.projector.set_viewpoints(self.metaByVid)

    def process(self, himages):
        self.hp_manager.process(himages)

        if False: # need reid between cameras
            # we estimate only one person per image
            hpsByPerson = {}
            if len(self.hp_manager.trackers) > 0:
                trackerkey = next(iter(self.hp_manager.trackers))
                tracker = self.hp_manager.trackers[trackerkey]
                if len(tracker.tracklets) > 0:
                    tracklet = next(iter(tracker.tracklets))
                    hps = tracker.tracklets[tracklet]
                    if len(hps) > 0:
                        # keep only one track (person) for now
                        print("project")
                        hpsForP1 = []
                        hpsForP1 = hps
                        # hpsByPerson["P1"] = hpsForP1  
                        hpsByPerson[tracklet] = hpsForP1  
                        skel3d = self.projector.process(hpsByPerson, self.hp_manager.estimator.minScoreKpt)
                        if skel3d is not None:
                            self.skels3d.append(skel3d)

    def print_stats(self):
        print("time estimation : {est_time.avg:.3f}\t".format(est_time=self.hp_manager.estim_time))
        print("time tracking : {est_time.avg:.3f}\t".format(est_time=self.hp_manager.tracking_time))


def run_pipeline(folder_path, config, use_tqdm=True):
    print(f"Estimage 2d pose for folder {os.path.dirname(folder_path)}")
    pipeline = Pipeline(config)

    # Collect videos info
    metaByVpath = {}
    maxIdxAll = 0
    for root, _, files in os.walk(folder_path):
        for f in files:
            if not f.endswith(('.mkv', '.mp4', '.avi')):
                continue

            fpath = os.path.join(root, f)
            meta = get_video_info(fpath)
            metaByVpath[fpath] = meta

            if meta['maxIdx'] > maxIdxAll:
                maxIdxAll = meta['maxIdx']

    # propagate camera information
    pipeline.set_viewpoints(metaByVpath)

    # process each image
    loading_time = AverageMeter()
    step = config.target_fps // config.sampling_by_sec
    indexes = range(0, maxIdxAll, step)

    if use_tqdm:
        indexes = tqdm(indexes)

    print("run on indexes: ", indexes)
    for idx in indexes:
        images = []
        for vidpath in metaByVpath:
            if idx < metaByVpath[vidpath]['maxIdx']:
                try:
                    st = time.time()
                    img = iio.imread(vidpath, index=idx)
                    loading_time.update(time.time() - st)
                    images.append(HImage(img, Path(vidpath), idx))
                except Exception as e:
                    print("failed reading image from ", vidpath, " at index #", idx)
                    print(e)
                    continue
        if len(images) == 0:
            break

        pipeline.process(images)

    print("time loading : {est_time.avg:.3f}\t".format(est_time=loading_time))
    pipeline.print_stats()

    return pipeline