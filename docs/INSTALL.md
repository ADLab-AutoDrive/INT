## Installation
Modified from [det3d](https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)'s original document.

### Requirements

- Linux
- Python 3.6+
- PyTorch 1.1 or higher
- CUDA 10.0 or higher
- CMake 3.13.2 or higher
- [APEX](https://github.com/nvidia/apex)
- [spconv](https://github.com/traveller59/spconv/commit/73427720a539caf9a44ec58abe3af7aa9ddb8e39) 

#### Notes
- spconv should be the specific version from links above
- The spconv version after this commit will consume much more memory. 
- A rule of thumb is that your pytorch cuda version must match the cuda version of your systsem for other cuda extensions to work properly. 

we have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04
- Python: 3.6.5
- PyTorch: 1.1
- CUDA: 10.0
- CUDNN: 7.5.0

### Basic Installation 

```bash
# basic python libraries
conda create --name centerpoint python=3.6
conda activate centerpoint
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
git clone https://github.com/ADLab-AutoDrive/INT
cd INT
pip install -r requirements.txt

# add INT to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_INT"
```

#### attention
may encounter cv2 problem when exec in docker:
```angular2html
    from .cv2 import *
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
this is because opencv installed by `pip uninstall opencv-python` can only be used locally.
if you want to use cv in docker, pls install as `pip install opencv-python-headless`.


### Advanced Installation 

#### Cuda Extensions

```bash
# set the cuda path(change the path to your own cuda location) 
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_PATH=/usr/local/cuda-10.0
export CUDA_HOME=/usr/local/cuda-10.0
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
bash setup.sh 
```

#### APEX

```bash
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 5633f6  # recent commit doesn't build in our system 
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### spconv
```bash
sudo apt-get install libboost-all-dev
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv && git checkout 7342772
python setup.py bdist_wheel
cd ./dist && pip install *
```
**error:** 
`CMake 3.13 or higher is required.  You are running version 3.5.1`

upgrade cmake by: 
```
apt remove cmake
pip install cmake --upgrade
ln -s /opt/conda/bin/cmake /usr/bin/cmake
```

**error:**
`spconv/third_party/pybind11
does not contain a CMakeLists.txt file.`

`git submodule init && git submodule update`

**error:**
`spconv/src/spconv/all.cc:20:91: error: no matching function for call to ‘torch::jit::RegisterOperators::RegisterOperators`

modify `src/spconv/all.cc`, remove `jit` namespace.
`torch::jit::RegisterOperators` to `torch::RegisterOperators`



#### tensorflow
```bash
pip3 install waymo-open-dataset-tf-2-6-0 --user

```



#### Check out [GETTING_START](GETTING_START.md) to prepare the data and play with all those pretrained models. 