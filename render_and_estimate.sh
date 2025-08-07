echo "Render and estimate meshes in folder : $1"

python tools/rendering/render.py $1 --convention='LSP' --method="PYRENDER" 

python estimate_pose_2d.py $1