import os
import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

samples_per_gpu = 4
workers_per_gpu = 8
nsweeps = 1

fm_channels = 256

voxel_size = [0.1, 0.1, 0.15]
pc_range = [-75.2, -75.2, -2, 75.2, 75.2, 4]

ds_factor = 8

full_WH = [round((x - y) / s) for x, y, s in zip(pc_range[3:5], pc_range[0:2], voxel_size[:2])]
fm_WH = [round(l / ds_factor) for l in full_WH]

# fm_fusion_method = None
cur_mid_channels = fm_channels // 2
past_mid_channels = fm_channels // 2
fm_fusion_method = dict(
    type="concat",  # ['concat', 'add', 'none']
    sub_type=None,
    input_channels=fm_channels,
    fm_WH=fm_WH,
    nsweeps=nsweeps,
    cur_pre_fusion_conv=dict(inplanes=fm_channels, planes=cur_mid_channels, num_blocks=1),
    past_pre_fusion_conv=dict(inplanes=fm_channels, planes=past_mid_channels, num_blocks=1),
    post_fusion_conv=dict(inplanes=cur_mid_channels + past_mid_channels, planes=fm_channels,
                          num_blocks=0),
    spatial_weight_conv=None,
)

pc_channels = 5
pc_channels += 1  # + timestamp
pc_fusion_method = dict(
    max_points_total=50000,
    max_points_per_frame=5000,
    min_points_per_frame=1000,
    max_time_limit=10,  # 10 seconds
    pc_channel=pc_channels,
    time_interval=0.1,
    keep_strategy="point_seg",  # ['point_seg', "in_box", "distance_random"]
    point_seg_thre=0.3,  # point seg score large than this value will be kept
)

tasks = [
    dict(num_class=3, class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST']),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))
common_heads = {'reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 'rot': (2, 2),
                'iou': (1, 2)}  # (output_channel, num_conv)
heads_chan = sum([v[0] for _, v in common_heads.items()])
hm_ds_factor = 1
hm_WH = [round(x / hm_ds_factor) for x in fm_WH]
hm_WHC = hm_WH + [len(class_names) + heads_chan * len(tasks)]
hm_fusion_method = dict(
    type="infinite_add",  # ['concat', 'add', 'none']
    sub_type="weighted",  # ['weighted', 'learned']
    use_reg=True,
    hm_WHC=hm_WHC,
    cur_pre_fusion_conv=None,
    past_pre_fusion_conv=None,
    post_fusion_conv=None,
    spatial_weight_conv=None,
)

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type='TwoStageDetector',
    first_stage_cfg=dict(
        type="VoxelNet",
        pretrained='path_to_your_1stage_checkpoint',
        fm_fusion_method=fm_fusion_method,
        hm_fusion_method=hm_fusion_method,
        pc_fusion_method=pc_fusion_method,
        batch_size=samples_per_gpu,
        freeze_backbone=False,
        reader=dict(
            type="DynamicVoxelEncoder",
            pc_range=pc_range,
            voxel_size=voxel_size,
        ),
        backbone=dict(
            type="SpMiddleFHD", num_input_features=pc_channels, ds_factor=ds_factor),
        neck=dict(
            type="RPN",
            layer_nums=[5, 5],
            ds_layer_strides=[1, 2],
            ds_num_filters=[128, 256],
            us_layer_strides=[1, 2],
            us_num_filters=[256, 256],
            num_input_features=fm_channels,
            logger=logging.getLogger("RPN"),
        ),
        bbox_head=dict(
            type="CenterHead",
            in_channels=sum([256, 256]),
            tasks=tasks,
            dataset='waymo',
            weight=2,
            iou_weight=1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            common_heads=common_heads,  # (output_channel, num_conv)
            hm_fusion_method=hm_fusion_method,
            point_seg={
                'point_features': pc_channels,
                'point_lift_mlp': [32, 64],
                'point_head_mlp': [64, 128],
                'pc_range': pc_range,
                'point_seg_loss_weight': 1.0,
            }
        ),
    ),
    second_stage_modules=[
        dict(
            type="BEVFeatureExtractor",
            pc_start=pc_range[:2],
            voxel_size=voxel_size[:2],
            out_stride=ds_factor
        )
    ],
    roi_head=dict(
        type="RoIHead",
        input_channels=(64 + hm_WHC[-1]) * 5 + 8,
        model_cfg=dict(
            CLASS_AGNOSTIC=True,
            SHARED_FC=[256, 256],
            CLS_FC=[256, 256],
            REG_FC=[256, 256],
            DP_RATIO=0.3,

            TARGET_CONFIG=dict(
                ROI_PER_IMAGE=128,
                FG_RATIO=0.5,
                SAMPLE_ROI_BY_EACH_CLASS=True,
                CLS_SCORE_TYPE='roi_iou',
                CLS_FG_THRESH=0.75,
                CLS_BG_THRESH=0.25,
                CLS_BG_THRESH_LO=0.1,
                HARD_BG_RATIO=0.8,
                REG_FG_THRESH=0.55
            ),
            LOSS_CONFIG=dict(
                CLS_LOSS='BinaryCrossEntropy',
                REG_LOSS='L1',
                LOSS_WEIGHTS={
                    'rcnn_cls_weight': 1.0,
                    'rcnn_reg_weight': 1.0,
                    'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                }
            )
        ),
        code_size=7
    ),
    NMS_POST_MAXSIZE=500,
    num_point=5,
    freeze=True
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    pc_range=pc_range,
    voxel_size=voxel_size,
    make_point_seg_label=True,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=4096,
        nms_post_max_size=500,
        nms_iou_threshold=0.7,
    ),
    score_threshold=0.03,
    pc_range=pc_range[:2],
    out_size_factor=get_downsample_factor(model),
    voxel_size=list(voxel_size[:2]),
    cf_weight=2
)

# dataset settings
dataset_type = "WaymoDataset"
data_root = "data/Waymo"
info_root = data_root

db_info_path = os.path.join(info_root, "dbinfos_train_seq_withvelo.pkl")
db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path=db_info_path,
    sample_groups=[
        dict(VEHICLE=15),
        dict(PEDESTRIAN=10),
        dict(CYCLIST=10),
    ],
    db_prep_steps=[],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
    seq_gt_db=True,
)

# db_sampler = None

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.95, 1.05],
    global_translate_std=0.5,
    db_sampler=db_sampler,
    class_names=class_names,
    no_augmentation=False,
    batch_size=samples_per_gpu,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = os.path.join(info_root, "infos_train_seq_filter_zero_gt.pkl")
val_anno = os.path.join(info_root, "infos_val_seq_filter_zero_gt.pkl")
test_anno = os.path.join(info_root, "infos_test_seq_filter_zero_gt.pkl")

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        ema_exp=True,
        pc_range=pc_range,
        sampled_interval=1,
        use_seq_info=True,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        ema_exp=True,
        pc_range=pc_range,
        use_seq_info=True,
        sampled_interval=1,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        test_mode=True,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        ema_exp=True,
        pc_range=pc_range,
        use_seq_info=True,
        sampled_interval=1,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1, out_dir='/model')
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook', log_dir=None)
    ],
)
# yapf:enable
# runtime settings
total_epochs = 6
max_training_seq = 100
training_seq_len = 100  # ['None' or int], if int is set, max_training_seq is invalid.
test_seq_len = 100
repeat_test = False
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
# work_dir = '/result/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None
workflow = [('train_infinite', 1)]
