import time
import os
from pathlib import Path

from .pipeline import Pipeline
from SensorData.HImage import HImage
from SensorData.utils import get_video_info
from settings.utils import AverageMeter
import imageio.v3 as iio
from tqdm import tqdm


def run_pipeline(folder_path, config, use_tqdm=True, plugin="FFMPEG"):
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
    # plugin = "pyav" if config.disp is False else "FFMPEG" # pyav freezes opencv imshow, so use "FFMPEG"
    for idx in indexes:
        images = []
        for vidpath in metaByVpath:
            if idx < metaByVpath[vidpath]['maxIdx']:
                try:
                    st = time.time()
                    img = iio.imread(vidpath, index=idx, plugin=plugin)
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