import imageio
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser("Read multiple videos and track people inside")
    parser.add_argument("videos_folder_path", type=str, help="Path to the folder where the videos are stored")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    for root, _, files in os.walk(args.videos_folder_path):
        for f in files:
            if not f.endswith(('.mkv', '.mp4', '.avi')):
                continue

            fpath = os.path.join(root, f)
            vid = imageio.get_reader(fpath, "ffmpeg")
            meta_data = vid.get_meta_data()
            print(meta_data)


