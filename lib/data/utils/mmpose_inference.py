
import os
from mmpose.apis import MMPoseInferencer

def run_mmpose_inference(img_dir, output_dir):
    inferencer = MMPoseInferencer('wholebody')

    for fname in os.listdir(img_dir):
        if '_rgba' in fname and fname.endswith('.png'):
            img_path = os.path.join(img_dir, fname)
            result_gen = inferencer(img_path, out_dir=output_dir)
            _ = next(result_gen)

    del inferencer

def run_mmpose_inference_batch(svd_dir, fidxs):
    inferencer = MMPoseInferencer('wholebody')

    input_paths = []

    # Collect all image paths across all fidxs
    for fid in fidxs:
        rescaled_rgba_dir = os.path.join(svd_dir, fid, 'rgba')
        mmpose_out_dir = os.path.join(svd_dir, fid, 'mmpose')
        os.makedirs(mmpose_out_dir, exist_ok=True)

        for fname in os.listdir(rescaled_rgba_dir):
            if fname.endswith('.png'):
                input_paths.append({
                    "img": os.path.join(rescaled_rgba_dir, fname),
                    "out_dir": mmpose_out_dir
                })

    # Run inference per image with correct output path
    for item in input_paths:
        result_gen = inferencer(item["img"], out_dir=item["out_dir"])
        try:
            next(result_gen)
        except StopIteration:
            pass
    del inferencer

