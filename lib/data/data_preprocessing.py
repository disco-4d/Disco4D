from lib.data.utils.setup_directories import setup_directories
from lib.data.utils.process_rgba import process_images_with_background_removal
from lib.data.utils.sv3d_inference import run_sv3d_inference
from lib.data.utils.lgm_inference import run_lgm_pipeline
from lib.data.utils.smplerx_inference import run_smplerx_inference
from lib.data.utils.mmpose_inference import run_mmpose_inference, run_mmpose_inference_batch
from lib.data.utils.config_generator import generate_configs
from lib.data.utils.segformer_segmentation import run_segformer_segmentation, run_segformer_segmentation_batch
from lib.data.utils.create_skin_images import generate_skin_variants
import os
import argparse

def run_pipeline(args):
    # Setup all working directories
    dirs = setup_directories(args.path, args.process_rgba)

    # Step 1: Background removal and recentering
    if args.process_rgba:
        process_images_with_background_removal(args.path, dirs['img_dir'], args.size, args.border_ratio, args.recenter, args.model)

    # Step 2: SV3D image generation and background removal per frame
    run_sv3d_inference(dirs['img_dir'], dirs['svd_dir'])

    # Step 3: LGM mesh reconstruction pipeline
    run_lgm_pipeline(dirs['img_dir'], dirs['svd_dir'], dirs['lgm_dir'])

    # Step 4: SMPLer-X body fitting
    run_smplerx_inference(dirs['img_dir'], dirs['smplx_dir'])

    # Step 5: MMPose inference for keypoints
    run_mmpose_inference(dirs['img_dir'], dirs['mmpose_dir'])
    fidxs = [fname.replace('.png', '') for fname in os.listdir(dirs['img_dir']) if fname.endswith('.png')]
    run_mmpose_inference_batch(dirs['svd_dir'], fidxs) 

    # Step 6: YAML config creation
    generate_configs(dirs['img_dir'], dirs['configs_dir'])

    # Step 7: Clothing segmentation
    run_segformer_segmentation(dirs['img_dir'], dirs['seg_dir'])
    run_segformer_segmentation_batch(dirs['svd_dir'], fidxs)

    # Step 8: Create clothless & average-skin RGB variants
    generate_skin_variants(dirs['img_dir'], dirs['seg_dir'], dirs['clothless_img_dir'], dirs['skin_img_dir'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image directory")
    parser.add_argument('--model', default='u2net', type=str)
    parser.add_argument('--size', default=1024, type=int)
    parser.add_argument('--process_rgba', action='store_true')
    parser.add_argument('--border_ratio', default=0.23, type=float)
    parser.add_argument('--recenter', action='store_true')
    args = parser.parse_args()

    run_pipeline(args)
