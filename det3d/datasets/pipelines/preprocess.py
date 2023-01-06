import os
import numpy as np

from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES

from det3d.core.utils.transformations import inverse_rigid_trans, transform_xyz, rotz, transform_from_rot_trans

VIZ = int(os.environ.get('VIZ', 0))


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)

        self.mode = cfg.mode
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_std = cfg.get('global_translate_std', 0)
            self.class_names = cfg.class_names
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None

            self.npoints = cfg.get("npoints", -1)

        self.no_augmentation = cfg.get('no_augmentation', False)
        if self.no_augmentation:
            self.shuffle_points = False

        self.random_state_dict = {}
        self.batch_size = cfg.get('batch_size', None)

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["WaymoDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
        elif res["type"] in ["NuScenesDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
        else:
            raise NotImplementedError

        # if res["lidar"]["annotations"] is None:
        #     if self.shuffle_points:
        #         np.random.shuffle(points)
        #         ## ATTENTION: here REWRITE res["lidar"]["points"], from now on use res["lidar"]["points"] is enough!
        #         res["lidar"]["points"] = points
        #     return res, info

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            if anno_dict is None:
                gt_dict = {
                    "gt_boxes": np.array([]).reshape(-1, 9),
                    "gt_names": np.array([]),
                }
            else:
                gt_dict = {
                    "gt_boxes": anno_dict["boxes"],
                    "gt_names": np.array(anno_dict["names"]).reshape(-1),
                }

        if self.mode == "train" and not self.no_augmentation:

            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= self.min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            if self.db_sampler:
                if res["metadata"]["use_seq_info"] is True:
                    assert 'random_seed' in info
                    np.random.seed(info['random_seed'])
                    train_seq_len_and_cur_frame = info['train_seq_len_and_cur_frame']
                else:
                    train_seq_len_and_cur_frame = None

                sampled_dict = self.db_sampler.sample_all(
                    # res["metadata"]["image_prefix"],
                    res["metadata"]["info_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None,
                    train_seq_len_and_cur_frame=train_seq_len_and_cur_frame,
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    ## make velocity to 0.
                    if sampled_gt_boxes.shape[1] > 7:
                        sampled_gt_boxes[:, 6:8] = 0  # make vel to 0, TODO make it ignore in loss computation.
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )

                    # # TODO a tmp fix for original baseline, creat gt_database again to add time idx!!!
                    # if (res["type"] in ["NuScenesDataset"]) and (sampled_points.shape[1] == 5):
                    #     sampled_points = np.hstack(
                    #         (sampled_points, np.zeros((sampled_points.shape[0], 1), dtype=np.float32)))

                    # if VIZ:
                    #     sampled_points = np.hstack(
                    #         (sampled_points, np.ones((sampled_points.shape[0], 1), dtype=np.float32)))
                    #     points = np.hstack(
                    #         (points, np.zeros((points.shape[0], 1), dtype=np.float32)))

                    points = np.concatenate([sampled_points, points], axis=0)

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            # set random seed again, as in db_sampler, the numbers of random called may be different.
            if res["metadata"]["use_seq_info"] is True:
                assert 'random_seed' in info
                np.random.seed(info['random_seed'])

            gt_dict["gt_boxes"], points, res['metadata']['flip_xy'] \
                = prep.random_flip_both(gt_dict["gt_boxes"], points, return_flip_state=True)

            gt_dict["gt_boxes"], points, res["metadata"]["rotation_ang"] = prep.global_rotation(
                gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise, return_rotation_ang=True
            )

            gt_dict["gt_boxes"], points, res["metadata"]["noise_scale"] = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise, return_noise_scale=True,
            )

            gt_dict["gt_boxes"], points, res["metadata"]["noise_trans"] = prep.global_translate_(
                gt_dict["gt_boxes"], points, noise_translate_std=self.global_translate_std, return_noise_trans=True,
            )

        elif self.no_augmentation:
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

        if self.shuffle_points:
            np.random.shuffle(points)

        ## ATTENTION: here REWRITE res["lidar"]["points"], from now on use res["lidar"]["points"] is enough!
        res["lidar"]["points"] = points

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        np.random.seed(None)

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num,
                                                                                  int) else cfg.max_voxel_num

        self.double_flip = cfg.get('double_flip', False)

        self.split_sweeps = cfg.get('split_sweeps', False)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def __call__(self, res, info):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        num_features = res['metadata']['num_point_features']

        if (res["mode"] == "train") and (res["lidar"]["annotations"] is not None):
            gt_dict = res["lidar"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]
            mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)

            res["lidar"]["annotations"] = gt_dict
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]

        if self.split_sweeps: # TODO refactor: remove
            assert res["mode"] in ["train", "val"], res['mode']
            points = res["lidar"]["points"]
            transform_matrix_list = res["lidar"]["transform_matrix_list"]
            assert points.shape[1] == 6, points.shape  # x y z intensity time_lag time_idx
            nsweeps = res['lidar']['nsweeps']
            sweep_idxs = res['lidar']['sweep_idxs']
            assert len(sweep_idxs) == nsweeps

            points_list = []
            for i, sweep_idx in enumerate(sweep_idxs):
                seq_mask = (points[:, -1] == sweep_idx)
                single_points = points[seq_mask]
                # transform back to original coords
                transform_matrix = transform_matrix_list[i]
                # if i > 0 and transform_matrix is None:
                #     # just a check, if is None, that is 10 repeated first frames!
                #     assert sum([x is None for x in transform_matrix_list]) == nsweeps
                # if transform_matrix is not None:
                # inv_transform_matrix = inverse_rigid_trans(transform_matrix[0:3, :])
                single_points[:, 0:3] = transform_xyz(single_points[:, 0:3], transform_matrix)
                points_list.append(single_points)

            voxels_list = []
            coords_list = []
            num_points_list = []
            num_voxels_list = []
            for single_points in points_list:
                voxels, coordinates, num_points = self.voxel_generator.generate(
                    single_points[:, 0:num_features],  # only use x y z intensity
                    max_voxels=max_voxels
                )
                voxels_list.append(voxels)
                coords_list.append(coordinates)
                num_points_list.append(num_points)
                num_voxels_list.append(np.array([voxels.shape[0]], dtype=np.int64))

            res["lidar"]["voxels"] = dict(
                voxels_list=voxels_list,
                coords_list=coords_list,
                num_points_list=num_points_list,
                num_voxels_list=num_voxels_list,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )


        else:
            voxels, coordinates, num_points = self.voxel_generator.generate(
                res["lidar"]["points"][:, 0:num_features], max_voxels=max_voxels
            )
            num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

            res["lidar"]["voxels"] = dict(
                voxels=voxels,
                coordinates=coordinates,
                num_points=num_points,
                num_voxels=num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

        double_flip = self.double_flip and (res["mode"] != 'train')

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

        return res, info


def flatten(box):
    return np.concatenate(box, axis=0)


def merge_multi_group_label(gt_classes, num_classes_by_task):
    num_task = len(gt_classes)
    flag = 0

    for i in range(num_task):
        gt_classes[i] += flag
        flag += num_classes_by_task[i]

    return flatten(gt_classes)


@PIPELINES.register_module
class AssignLabel(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.cfg = assigner_cfg

        self.make_point_seg_label = assigner_cfg.get('make_point_seg_label', False)

    def __call__(self, res, info):
        max_objs = self._max_objs
        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]

        example = {}

        if (res["mode"]) == "train" and (res["lidar"]["annotations"] is not None):

            # Calculate output featuremap size
            try:
                grid_size = res["lidar"]["voxels"]["shape"]
                pc_range = res["lidar"]["voxels"]["range"]
                voxel_size = res["lidar"]["voxels"]["size"]
            except KeyError:
                pc_range = np.array(self.cfg["pc_range"])
                voxel_size = self.cfg["voxel_size"]
                grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
                grid_size = np.round(grid_size).astype(np.int64)

            feature_map_size = grid_size[:2] // self.out_size_factor

            gt_dict = res["lidar"]["annotations"]

            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

            draw_gaussian = draw_umich_gaussian

            hms, anno_boxs, inds, masks, cats = [], [], [], [], []

            for idx, task in enumerate(self.tasks):
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                              dtype=np.float32)

                if res['type'] == 'NuScenesDataset':
                    # [reg, hei, dim, vx, vy, rots, rotc]
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32)
                elif res['type'] == 'WaymoDataset':
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32)
                else:
                    raise NotImplementedError("Only Support nuScene for Now!")

                ind = np.zeros((max_objs), dtype=np.int64)
                mask = np.zeros((max_objs), dtype=np.uint8)
                cat = np.zeros((max_objs), dtype=np.int64)

                num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)

                for k in range(num_objs):
                    cls_id = gt_dict['gt_classes'][idx][k] - 1

                    w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], \
                              gt_dict['gt_boxes'][idx][k][5]
                    w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))

                        # be really careful for the coordinate system of your box annotation. 
                        x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], \
                                  gt_dict['gt_boxes'][idx][k][2]

                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                         (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32)
                        ct_int = ct.astype(np.int32)

                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                            continue

                        draw_gaussian(hm[cls_id], ct, radius)

                        new_idx = k
                        x, y = ct_int[0], ct_int[1]

                        cat[new_idx] = cls_id
                        ind[new_idx] = y * feature_map_size[0] + x
                        mask[new_idx] = 1

                        if res['type'] == 'NuScenesDataset':
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][8]
                            anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                 np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        elif res['type'] == 'WaymoDataset':
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][-1]
                            anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                 np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        else:
                            raise NotImplementedError("Only Support Waymo and nuScene for Now")

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)

            # used for two stage code 
            boxes = flatten(gt_dict['gt_boxes'])
            classes = merge_multi_group_label(gt_dict['gt_classes'], num_classes_by_task)

            if res["type"] == "NuScenesDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            elif res['type'] == "WaymoDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            else:
                raise NotImplementedError()

            boxes_and_cls = np.concatenate((boxes,
                                            classes.reshape(-1, 1).astype(np.float32)), axis=1)
            num_obj = len(boxes_and_cls)
            assert num_obj <= max_objs
            # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
            boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
            gt_boxes_and_cls[:num_obj] = boxes_and_cls

            example.update({'gt_boxes_and_cls': gt_boxes_and_cls})

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})

            # make_point_seg_label
            if self.make_point_seg_label:
                points = res['lidar']['points']
                point_indices = box_np_ops.points_in_rbbox(points[:, :3], boxes)
                points_seg_label = point_indices.sum(1).astype(np.float32)
                example['points_seg_label'] = points_seg_label

        else:
            pass

        res["lidar"]["targets"] = example

        return res, info
