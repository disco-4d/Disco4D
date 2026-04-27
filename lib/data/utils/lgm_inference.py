# lib/data/utils/lgm_inference.py

import os
import glob
from LGM.infer_dch_pipe2 import process
from LGM.convert_dch import Converter
from LGM.core.options import Options

def run_lgm_pipeline(img_dir, svd_dir, lgm_dir):
    ckpt_path = "../LGM/pretrained/model_fp16_fixrot.safetensors"
    opt = Options(
        input_size=256,
        up_channels=(1024, 1024, 512, 256, 128),
        up_attention=(True, True, True, False, False),
        splat_size=128,
        output_size=512,
        batch_size=8,
        num_views=8,
        gradient_accumulation_steps=1,
        mixed_precision='bf16',
        resume=ckpt_path,
        test_path=img_dir,
        workspace=lgm_dir
    )

    for fname in os.listdir(img_dir):
        fid = fname.split('.png')[0]
        if os.path.exists(os.path.join(lgm_dir, f'{fid}.ply')):
            continue
        img_paths = [os.path.join(svd_dir, fid, 'img', f'{i:04d}.png') for i in [0, 5, 10, 15]]
        process(opt, img_paths, fid)

    opt.test_path = lgm_dir
    for ply_file in glob.glob(f'{lgm_dir}/*.ply'):
        obj_file = ply_file.replace('.ply', '.obj')
        if os.path.exists(obj_file):
            continue
        print(f"Processing {ply_file}")
        opt.test_path = ply_file
        converter = Converter(opt).cuda()
        converter.fit_nerf()
        converter.fit_mesh()
        converter.fit_mesh_uv()
        converter.export_mesh(obj_file)
        del converter
