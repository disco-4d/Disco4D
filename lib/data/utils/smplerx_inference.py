
import os
import glob
import shutil
from smplerx.main.inference_dch import infer_dch

def run_smplerx_inference(img_dir, smplx_dir):
    infer_dch(img_dir, smplx_dir)

    dir_name = os.path.basename(os.path.dirname(img_dir))
    clean_path = f'data/{dir_name}/smplerx/smplx'
    for npz in glob.glob(f'{clean_path}/*.npz'):
        new_npz = npz.replace('.png_0', '')
        shutil.move(npz, new_npz)
