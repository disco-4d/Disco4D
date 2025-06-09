
import os
import glob
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

def run_segformer_segmentation(img_dir, seg_dir):
    os.makedirs(seg_dir, exist_ok=True)
    device = 'cuda'

    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(device)

    transform = T.ToPILImage()
    clothing_dict = {
        0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress",
        8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg",
        14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
    }
    cat_map = {16: 0, 17: 0}

    for file in glob.glob(f'{img_dir}/*.png'):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.astype(np.float32) / 255.0
        im = torch.from_numpy(img).to(device)

        image = transform(im.permute(2, 0, 1))
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = F.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]

        input_tensor = pred_seg.clone()
        for k, v in cat_map.items():
            input_tensor = torch.where(pred_seg == k, v, input_tensor)

        save_path = file.replace('_rgba.png', '_seg.pt').replace('/img/', '/seg/')
        if save_path != file:
            torch.save(input_tensor, save_path)

    del model
    del processor

def run_segformer_segmentation_batch(svd_dir, fidxs_list):
    device = 'cuda'
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(device)
    transform = T.ToPILImage()
    cat_map = {16: 0, 17: 0}

    for fidxs in fidxs_list:
        rgba_dir = os.path.join(svd_dir, fidxs, 'rgba')
        seg_dir = os.path.join(svd_dir, fidxs, 'seg')
        os.makedirs(seg_dir, exist_ok=True)

        for fname in os.listdir(rgba_dir):
            if not fname.endswith('.png'):
                continue
            img_path = os.path.join(rgba_dir, fname)

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
            im = torch.from_numpy(img).to(device)

            image = transform(im.permute(2, 0, 1))
            inputs = processor(images=image, return_tensors="pt").to(device)
            logits = model(**inputs).logits.cpu()

            upsampled_logits = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
            pred_seg = upsampled_logits.argmax(dim=1)[0]

            for k, v in cat_map.items():
                pred_seg = torch.where(pred_seg == k, v, pred_seg)

            save_path = os.path.join(seg_dir, fname.replace('.png', '_seg.pt'))
            torch.save(pred_seg, save_path)

    del model
    del processor