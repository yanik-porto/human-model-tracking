import argparse
import os
import glob
import concurrent.futures
import time

from settings.config import load_config
from Pipeline.runner_folder import run_pipeline_on_folder, run_pipeline_on_file

def parse_args():
    parser = argparse.ArgumentParser("Read multiple set of videos and save pose estimation")
    parser.add_argument("dataset_folder_path", type=str, help="Path to the folder where the dataset is stored")
    parser.add_argument("--overwrite", action="store_true", default=False, help="If set, overwrite existing estimation files")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to run in parallel")
    parser.add_argument("--by_file", action="store_true", default=False, help="If set, iterate by video files, instead of by folder")
    return parser.parse_args()

def run_pipeline_and_save_projections(folder_path, config, use_tqdm=True, by_file=False):
    if by_file:
        pipeline = run_pipeline_on_file(folder_path, config, use_tqdm=use_tqdm)#, plugin="pyav")
    else:
        pipeline = run_pipeline_on_folder(folder_path, config, use_tqdm=use_tqdm)
    pipeline.hp_manager.save_projections()

if __name__ == "__main__":
    args = parse_args()

    config = load_config()

    paths = []
    for root, folders, files in os.walk(args.dataset_folder_path):
        if args.by_file:
            for f in files:
                if not f.endswith(('.mkv', '.mp4', '.avi')):
                    continue

                file_path = os.path.join(root, f)
                npz_files = glob.glob(os.path.splitext(file_path)[0] + "*[0-9].npz")
                if len(npz_files) > 0:
                    if not args.overwrite:
                        print(f"Skip estimation for {f}, files exist yet")
                        continue
                    else:
                        print(f"Delete existing estimation files for {f}")
                        for npz_f in npz_files:
                            os.remove(npz_f)
                paths.append(file_path)

        else:                
            for folder in folders:
                folder_path = os.path.join(root, folder)

                if not args.overwrite:
                    npz_files = glob.glob(os.path.join(folder_path, "*[0-9].npz"))
                    if len(npz_files) > 0:
                        print(f"Skip estimation for {folder}, files exist yet")
                        continue 
                paths.append(folder_path)

    print(f"{len(paths)} folders to process")

    st = time.perf_counter()
    if args.num_workers == 1:
        for path in paths:
            run_pipeline_and_save_projections(path, config, use_tqdm=True, by_file=args.by_file)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futur_to_path = {executor.submit(run_pipeline_and_save_projections, path, config, False): path for path in paths}
            for future in concurrent.futures.as_completed(futur_to_path):
                path = futur_to_path[future]
                print(path, " completed successfully")
    print("time total : ", (time.perf_counter() - st) / 60.,  " min")