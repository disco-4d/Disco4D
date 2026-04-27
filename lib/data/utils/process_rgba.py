import os
import cv2
import numpy as np
import rembg

def process_images_with_background_removal(input_dir, output_dir, size, border_ratio, recenter, model_name):
    session = rembg.new_session(model_name=model_name)
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for file in image_files:
        file_path = os.path.join(input_dir, file)
        base_name = os.path.splitext(file)[0]
        out_path = os.path.join(output_dir, f"{base_name}_square_rgba.png")

        if os.path.exists(out_path):
            continue

        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        carved_image = rembg.remove(
            image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=254,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=5,
            post_process_mask=True
        )
        mask = carved_image[..., -1] > 0

        if recenter:
            final_rgba = np.zeros((size, size, 4), dtype=np.uint8)
            coords = np.nonzero(mask)
            x_min, x_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            h, w = x_max - x_min, y_max - y_min
            desired_size = int(size * (1 - border_ratio))
            scale = desired_size / max(h, w)
            h2, w2 = int(h * scale), int(w * scale)
            x2_min, y2_min = (size - h2) // 2, (size - w2) // 2
            x2_max, y2_max = x2_min + h2, y2_min + w2
            final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
                carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        else:
            final_rgba = carved_image

        cv2.imwrite(out_path, final_rgba)