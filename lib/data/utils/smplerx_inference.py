
import os
import glob
import shutil
from smplerx.main.inference_dch import infer_dch

def run_smplerx_inference(img_dir, smplx_dir):
    dir_name = os.path.basename(os.path.dirname(img_dir))
    clean_path = f'data/{dir_name}/smplerx/smplx'

    fids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.png')]
    if fids and all(os.path.exists(os.path.join(clean_path, f'{fid}.npz')) for fid in fids):
        return

    infer_dch(img_dir, smplx_dir)

    for npz in glob.glob(f'{clean_path}/*.npz'):
        new_npz = npz.replace('.png_0', '')
        shutil.move(npz, new_npz)
