# Install

conda env create --file environment.yml
conda activate hm
python -m pip install -r HPEstimation/HRNET/requirements.txt
python -m pip install -r HPEstimation/YOLO/requirements.txt
python -m pip install -r HM/SMPL/requirements.txt


# Run

## Estimtate poses from videos

python estimate_pose_2d.py <PATH_TO_FOLDER>

