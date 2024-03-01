import os
import imageio
from settings.utils import load_file_to_matrix

def get_video_info(fpath):
    vid = imageio.get_reader(fpath, "ffmpeg")
    meta_data = vid.get_meta_data()
    print(meta_data)

    fpathnoext, _ = os.path.splitext(fpath)
    campath = fpathnoext + '.txt'
    camP = load_file_to_matrix(campath)

    meta = {}
    meta['maxIdx'] = int(meta_data['duration'] * meta_data['fps'])
    meta['P'] = camP

    print(meta)
    return meta