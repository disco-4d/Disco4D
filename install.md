## Installation

```bash
conda create -n Disco4D python=3.8
conda activate Disco4D

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -q -y pyg -c pyg
conda install nvidia/label/cuda-11.8.0::cuda-toolkit

pip install ./diffusers
pip install -r requirements.txt

pip install ./diff-gaussian-rasterization
pip install ./simple-knn
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install git+https://github.com/ashawkey/kiuikit




# install lgm
# git clone https://github.com/3DTopia/LGM.git
pip install tyro
pip install https://download.pytorch.org/whl/cu118/xformers-0.0.22.post7%2Bcu118-cp38-cp38-manylinux2014_x86_64.whl
pip install roma
pip install nerfacc


# install mmcv
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
# git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
cd ..

# install smplerx
# git clone https://github.com/caizhongang/SMPLer-X
# mv SMPLer-X smplerx
pip install torchgeometry # edit it
pip install timm

pip install numpy==1.23.1
pip install pyopengl


pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html
pip install open3d

# knn cuda
git clone https://github.com/unlimblue/KNN_CUDA.git
cd KNN_CUDA
make && make install

# pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install rtree

```

Tested on:

- Ubuntu 22 with torch 2.1.0 & CUDA 11.8 on a 3090 and A6000.


#### FAQ
- `RuntimeError: Subtraction, the '-' operator, with a bool tensor is not supported. If you are trying to invert a mask, use the '~' or 'logical_not()' operator instead.`
  
  Follow [this post](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527) and modify `torchgeometry`

