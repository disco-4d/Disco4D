
CASE_DIR=Actors # teaser
WORK_DIR=$(pwd)

# python main2.py --config configs/image.yaml --dir ${CASE_DIR} force_cuda_rast=True radius=1.5 gui=False --batch True --num_frames 20 
# python lib/ops/mesh_to_gs.py --dir ${CASE_DIR}

python main.py --config configs/image_sv3d_pipe2.yaml --batch True --dir ${CASE_DIR} --stage 12 --num_frames 20

# python main_svd_smplx.py --config configs/4d.yaml --model disco4d_single --batch True --dir ${CASE_DIR} elevation=0 force_cuda_rast=True gui=False \
#     n_views=8 iters=1000 use_pretrained=False batch_size=1 ref_size=1024 --num_frames 20