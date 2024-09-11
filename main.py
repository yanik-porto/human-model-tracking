import argparse
import os
import sys
import numpy as np
import pickle

from settings.config import load_config
from Pipeline.runner_folder import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser("Read multiple videos and track people inside")
    parser.add_argument("videos_folder_path", type=str, help="Path to the folder where the videos are stored")
    parser.add_argument("--save_2d_proj", action="store_true", default=False, help="Set if 2d projections need to be saved")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    config = load_config()

    pipeline = run_pipeline(args.videos_folder_path, config)

    if args.save_2d_proj:
        pipeline.hp_manager.save_projections()

    sys.exit(0)

    with open("skel3d.pkl", "wb") as fsk:
        pickle.dump(pipeline.skels3d, fsk)


    timestamps = []

    for time in timestamps:
        # TODO : collect images at timestamp
        images = []


        action = pipeline.action_sensor.current_action
        n_persons = pipeline.tracker.n_persons

        print("current action: " + action)
        print("current number of persons: ", n_persons)
