import argparse
import os
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Compress a list of skeleton files into a ntu dataset format")
    parser.add_argument("in_path", type=str, help="Path to input folder")
    parser.add_argument("--unknown_label", action="store_true", default=False, help="set if the label is unknown")
    parser.add_argument("--only_suffix", type=str, default=".npz", help="Set if only files with the given suffix have to be taken into account")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    ntu_format = {}
    ntu_format["split"] = {}
    ntu_format["split"]["xsub_val"] = []
    ntu_format["annotations"] = []

    for root, _, files in os.walk(args.in_path):
        for f in files:
            if f.endswith(args.only_suffix):
                print(f)
                skel_path = os.path.join(root, f)
                skel = dict(np.load(skel_path))
                print(skel.keys())

                if "keypoint" not in skel:
                    continue

                annot = {}
                keypoint = skel["keypoint"]
                if len(keypoint.shape) == 3:
                    keypoint = np.expand_dims(keypoint, axis=0)
                annot["keypoint"] = keypoint

                if "keypoint_score" in skel:
                    keypoint_score = skel["keypoint_score"]
                    if len(keypoint_score.shape) == 2:
                        keypoint_score = np.expand_dims(keypoint_score, axis=0)
                    annot["keypoint_score"] = keypoint_score

                frame_dir = f.split(".")[0]
                if args.unknown_label:
                    annot["label"] = 0
                else:
                    splits = frame_dir.split("_")
                    if len(splits) == 1:
                        frame_dir = frame_dir[0]
                    elif len(splits) > 1:
                        frame_dir = frame_dir[:-2]
                    annot["label"] = int(frame_dir[-3:]) - 1
                
                annot["frame_dir"] = frame_dir
                annot["img_shape"] = (1080, 1920)
                annot["original_shape"] = (1080, 1920)
                annot["total_frames"] = keypoint.shape[1]

                ntu_format["annotations"].append(annot)
                ntu_format["split"]["xsub_val"].append(frame_dir)

    with open(os.path.join(args.in_path, "ntu_custom.pkl"), "wb") as outf:
        pickle.dump(ntu_format, outf)
    