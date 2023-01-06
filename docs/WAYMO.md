## Getting Started with CenterPoint on Waymo

### Prerequisite 

- Follow [INSTALL.md](INSTALL.md) to install all required libraries. 
- Tensorflow 
- Waymo-open-dataset devkit

```bash
conda activate centerpoint 
pip install waymo-open-dataset-tf-1-15-0==1.2.0 
```

### Prepare data

#### Download data and organise as follows

```
# For Waymo Dataset         
└── WAYMO_DATASET_ROOT
       ├── tfrecord_training       
       ├── tfrecord_validation   
       ├── tfrecord_testing 
```

Convert the tfrecord data to pickle files.

```bash
# train set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'WAYMO_DATASET_ROOT/tfrecord_training/*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/train/'

# validation set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'WAYMO_DATASET_ROOT/tfrecord_validation/*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/val/'

# testing set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'WAYMO_DATASET_ROOT/tfrecord_testing/*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/test/'
```

Create a symlink to the dataset root 
```bash
mkdir data && cd data
ln -s WAYMO_DATASET_ROOT Waymo
```
Remember to change the WAYMO_DATASET_ROOT to the actual path in your system. 


#### Create info files

```bash
# Sequence-wise Infos 
python tools/create_data.py waymo_data_prep_seqwise --root_path=data/Waymo --split train

python tools/create_data.py waymo_data_prep_seqwise --root_path=data/Waymo --split val

python tools/create_data.py waymo_data_prep_seqwise --root_path=data/Waymo --split test
```

In the end, the data and info files should be organized as follows

```
└── CenterPoint
       └── data    
              └── Waymo 
                     ├── tfrecord_training       
                     ├── tfrecord_validation
                     ├── train <-- all training frames and annotations 
                     ├── val   <-- all validation frames and annotations 
                     ├── test   <-- all testing frames and annotations 
                     ├── infos_train_seq_filter_zero_gt.pkl
                     ├── infos_val_seq_filter_zero_gt.pkl
                     ├── infos_test_seq_filter_zero_gt.pkl
                     ├── dbinfos_train_seq_withvelo.pkl
                     ├── gt_database_seq_withvelo
```

### Train & Evaluate in Command Line

Use the following command to start a distributed training using 4 GPUs. The models and logs will be saved to ```work_dirs/CONFIG_NAME```. 

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_infinte.py --config CONFIG_PATH
```

For distributed testing with 4 gpus,

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test_infinite.py --config CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth 
```

For testing with one gpu and see the inference time,

```bash
python ./tools/dist_test_infinite.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --speed_test 
```

This will generate a `my_preds.bin` file in the work_dir. You can create submission to Waymo server using waymo-open-dataset code by following the instructions [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md).  


### Test Set 

Add the ```--testset``` flag to the end. 

```bash
python ./tools/dist_test.py --config CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --testset 
```