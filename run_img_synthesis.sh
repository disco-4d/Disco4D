#!/bin/bash

CASE_DIR=4ddress_img
WORK_DIR=$(pwd)
export PYTHONPATH=$(pwd)

# Preprocess data
python lib/data/data_preprocessing.py data/${CASE_DIR}/img 

# Optimize SMPLX GS
python lib/mesh_utils/main_smplx.py --dir ${CASE_DIR} 

# Visual hull 
python lib/ops/visual_hull.py --dir ${CASE_DIR} 

# Optimize gaussians
python lib/gs_utils/main_smplx.py --dir ${CASE_DIR} 
