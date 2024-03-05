import argparse
import os
import glob

from settings.config import load_config
from Pipeline.pipeline import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser("Read multiple set of videos and save pose estimation")
    parser.add_argument("dataset_folder_path", type=str, help="Path to the folder where the dataset is stored")
    parser.add_argument("--overwrite", action="store_true", default=False, help="If set, overwrite existing estimation files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    config = load_config()

    for root, folders, files in os.walk(args.dataset_folder_path):
        for folder in folders:
            folder_path = os.path.join(root, folder)

            if not args.overwrite:
                npz_files = glob.glob(os.path.join(folder_path, "*.npz"))
                if len(npz_files) > 0:
                    print(f"Skip estimation for {folder}, files exist yet")
                    continue 

            print(f"Estimage 2d pose for folder {folder}")

            pipeline = run_pipeline(folder_path, config)

            pipeline.hp_manager.save_projections()
