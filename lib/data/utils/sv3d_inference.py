import os
import numpy as np
import torch
import imageio
import cv2
from PIL import Image
import rembg

from guidance.sv3d import build_sv3d_model, sample as sv3d_pipe

def _sv3d_done(svd_dir, fid):
    mp4 = os.path.join(svd_dir, f"{fid}_sv3d.mp4")
    rgba_0 = os.path.join(svd_dir, fid, 'rgba', '0000.png')
    img_20 = os.path.join(svd_dir, fid, 'img', '0020.png')
    return os.path.exists(mp4) and os.path.exists(rgba_0) and not os.path.exists(img_20)


def run_sv3d_inference(img_dir, svd_dir):
    elevations_deg = 0
    azimuths_deg = list(np.linspace(0, 360, 21) % 360)

    pending = [fname for fname in os.listdir(img_dir)
               if not _sv3d_done(svd_dir, os.path.splitext(fname)[0])]
    if not pending:
        return

    model = build_sv3d_model(num_steps=30, device='cuda')

    for fname in pending:
        file_path = os.path.join(img_dir, fname)
        fid = os.path.splitext(fname)[0]
        frames = sv3d_pipe(
            model=model,
            input_path=file_path,
            version='sv3d_p',
            elevations_deg=elevations_deg,
            azimuths_deg=azimuths_deg
        )

        imageio.mimwrite(os.path.join(svd_dir, f"{fid}_sv3d.mp4"), frames)

        svd_img_dir = os.path.join(svd_dir, fid, 'img')
        os.makedirs(svd_img_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(svd_img_dir, f"{i:04d}.png"))

        torch.cuda.empty_cache()

        # Background removal
        session = rembg.new_session(model_name='u2net')
        svd_rgba_dir = os.path.join(svd_dir, fid, 'rgba')
        os.makedirs(svd_rgba_dir, exist_ok=True)

        for img_name in os.listdir(svd_img_dir):
            img_path = os.path.join(svd_img_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            rgba = rembg.remove(img, session=session,
                                alpha_matting=True,
                                alpha_matting_foreground_threshold=254,
                                alpha_matting_background_threshold=20,
                                alpha_matting_erode_size=5,
                                post_process_mask=True)
            # cv2.imwrite(os.path.join(svd_rgba_dir, img_name), rgba)
            # Convert to PIL Image for resizing
            img_pil = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))

            # Resize to 1024x1024
            img_rescaled = img_pil.resize((1024, 1024), Image.LANCZOS)

            # Save image
            out_path = os.path.join(svd_rgba_dir, img_name)
            img_rescaled.save(out_path)


        os.rename(os.path.join(svd_img_dir, '0020.png'), os.path.join(svd_img_dir, '0000.png'))
        os.rename(os.path.join(svd_rgba_dir, '0020.png'), os.path.join(svd_rgba_dir, '0000.png'))
