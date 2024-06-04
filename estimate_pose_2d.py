import argparse
import os
import glob
import concurrent.futures

from settings.config import load_config
from Pipeline.pipeline import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser("Read multiple set of videos and save pose estimation")
    parser.add_argument("dataset_folder_path", type=str, help="Path to the folder where the dataset is stored")
    parser.add_argument("--overwrite", action="store_true", default=False, help="If set, overwrite existing estimation files")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to run in parallel")
    return parser.parse_args()

def run_pipeline_and_save_projections(folder_path, config, use_tqdm=True):
    pipeline = run_pipeline(folder_path, config, use_tqdm=use_tqdm)
    pipeline.hp_manager.save_projections()

if __name__ == "__main__":
    args = parse_args()

    config = load_config()

    paths = []
    for root, folders, files in os.walk(args.dataset_folder_path):
        for folder in folders:
            folder_path = os.path.join(root, folder)

            if not args.overwrite:
                npz_files = glob.glob(os.path.join(folder_path, "*[0-9].npz"))
                if len(npz_files) > 0:
                    print(f"Skip estimation for {folder}, files exist yet")
                    continue 

            paths.append(folder_path)

    if args.num_workers == 1:
        for path in paths:
            run_pipeline_and_save_projections(path, config, use_tqdm=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futur_to_path = {executor.submit(run_pipeline_and_save_projections, path, config, False): path for path in paths}
            for future in concurrent.futures.as_completed(futur_to_path):
                path = futur_to_path[future]
                print(path, " completed successfully")
