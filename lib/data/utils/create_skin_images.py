
import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

def generate_skin_variants(img_dir, seg_dir, clothless_dir, skincolor_dir):
    parsed_cloth_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    parsed_skin_idxs = [11, 12, 13, 14, 15]
    os.makedirs(clothless_dir, exist_ok=True)
    os.makedirs(skincolor_dir, exist_ok=True)

    for fname in tqdm(sorted(glob.glob(f'{img_dir}/*.png'))):
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        alpha = img[:, :, -1]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0

        seg_file = fname.replace('_rgba.png', '_seg.pt').replace('/img/', '/seg/')
        seg = torch.load(seg_file).cpu().numpy()

        def extract_mask(seg, img, indexes):
            mask = np.isin(seg, indexes).astype(np.uint8)
            mask[mask > 0] = 1
            mask_3ch = np.repeat(mask[:, :, None], 3, axis=2)
            return np.stack([mask * 255]*3, axis=-1), mask, img * mask_3ch

        nonskin_rgb, nonskin_mask, nonskin_img = extract_mask(seg, img, parsed_cloth_idxs)
        skin_rgb, skin_mask, skin_img = extract_mask(seg, img, parsed_skin_idxs)

        skin_mask_bool = skin_mask.astype(bool)
        nonskin_mask_bool = nonskin_mask.astype(bool)
        np_img_uint8 = (img * 255).astype(np.uint8)
        masked_skin = cv2.bitwise_and(np_img_uint8, np_img_uint8, mask=skin_mask)

        rgb_vals = masked_skin.reshape(-1, 3)
        rgb_vals = [tuple(rgb) for rgb in rgb_vals if not np.array_equal(rgb, [0, 0, 0])]
        avg_rgb = np.array(rgb_vals).mean(0) if rgb_vals else [128, 128, 128]

        filled = np.full_like(np_img_uint8, avg_rgb)
        result = filled.copy()
        result[skin_mask_bool] = np_img_uint8[skin_mask_bool]
        rgba = np.dstack((result, np.ones_like(alpha) * 255))
        Image.fromarray(rgba.astype(np.uint8)).save(fname.replace('/img/', '/clothless_img/'))

        full_skin = np.dstack((filled, np.ones_like(alpha) * 255))
        Image.fromarray(full_skin.astype(np.uint8)).save(fname.replace('/img/', '/skin_img/'))
