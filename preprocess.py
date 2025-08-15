from mesh import Mesh
from glob import glob
import os
import shutil

os.makedirs('logs/output/Actors')
base_dir = "data/Actors"

for images_dir in glob(os.path.join(base_dir, "*/images")):
    dest_dir = os.path.join(os.path.dirname(images_dir), "train_images")
    os.makedirs(dest_dir, exist_ok=True)

    for file_path in glob(os.path.join(images_dir, "train*")):
        if os.path.isfile(file_path):
            shutil.copy2(file_path, dest_dir)  
            print(f"Copied: {file_path} -> {dest_dir}")

for seg_dir in glob(os.path.join(base_dir, "*/segmentation_masks")):
    dest_dir = os.path.join(os.path.dirname(seg_dir), "train_segmentation_masks")
    os.makedirs(dest_dir, exist_ok=True)

    for file_path in glob(os.path.join(seg_dir, "train*")):
        if os.path.isfile(file_path):
            shutil.copy2(file_path, dest_dir) 
            print(f"Copied: {file_path} -> {dest_dir}")


ply_files = glob('data/Actors/*/smplx.ply')

for ply_file in ply_files:
    basename = os.path.splitext(ply_file.split('/')[-2])[0]
    mesh_ply = Mesh.load(ply_file, resize=False)
    mesh = Mesh.load("data/smplx_uv/smplx_tex.obj", albedo_path='data/smplx_uv/blank_albedo.png')
    mesh.v = mesh_ply.v
    mesh.write_obj(f'logs/output/Actors/{basename}_smplx.obj')