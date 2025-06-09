CASE_DIR=4ddress_video
CASE_IMG_DIR=4ddress_img
WORK_DIR=$(pwd)
export PYTHONPATH=$(pwd)

python lib/gs_4d_utils/main_4d_smplx.py --config configs/4d.yaml --model disco4d_smplx --dir ${CASE_DIR} force_cuda_rast=True radius=1.5 gui=False --batch True n_views=2 iters=2000 use_pretrained=False --img_dir ${CASE_IMG_DIR}
