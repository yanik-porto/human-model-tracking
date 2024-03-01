import argparse
import os

from settings.config import load_config
from Pipeline.pipeline import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser("Read multiple set of videos and save pose estimation")
    parser.add_argument("dataset_folder_path", type=str, help="Path to the folder where the dataset is stored")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    config = load_config()

    for root, folders, files in os.walk(args.dataset_folder_path):
        for folder in folders:
            print(f"Estimage 2d pose for folder {folder}")

            folder_path = os.path.join(root, folder)

            pipeline = run_pipeline(folder_path, config)

            pipeline.hp_manager.save_projections()
