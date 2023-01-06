import os.path as osp
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle
import os
from ..registry import PIPELINES
from ...core.utils.transformations import inverse_rigid_trans


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def read_file(path, tries=2, num_point_feature=4, painted=False):
    if painted:
        dir_path = os.path.join(*path.split('/')[:-2], 'painted_' + path.split('/')[-2])
        painted_path = os.path.join(dir_path, path.split('/')[-1] + '.npy')
        points = np.load(painted_path)
        points = points[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]  # remove ring_index from features
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, painted=False, remove_sweep_close_points=True):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), painted=painted).T
    if remove_sweep_close_points:
        points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T


def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)

    return points


def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T  # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T


def get_obj(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)
        self.remove_sweep_close_points = kwargs.get("remove_sweep_close_points", True)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]
            # transform_point_cloud = res["lidar"]["transform_point_cloud"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res.get("painted", False))

            if nsweeps == 1:
                res["lidar"]["points"] = points
                res["lidar"]["transform_matrix"] = info.get("transform_matrix", None)

            else:
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]
                transform_matrix_list = [np.eye(4, dtype=np.float64)[0:3]]  # list of 3x4 transform matrix
                # transform_matrix_list = [None]
                sweep_idx_list = [np.zeros((points.shape[0], 1))]
                sweep_idx_list2 = [0]
                time_lag_list = [0]

                assert (nsweeps - 1) <= len(
                    info["sweeps"]
                ), "nsweeps {} should equal to or less than list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                choosed_sweeps = sorted(np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False))
                for i in choosed_sweeps:
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_sweep(sweep, painted=res.get("painted", False),
                                                           remove_sweep_close_points=self.remove_sweep_close_points)
                    # transform_point_cloud=transform_point_cloud)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)
                    sweep_idx_list.append((i + 1) * np.ones((points_sweep.shape[0], 1)))
                    sweep_idx_list2.append(i + 1)
                    if sweep["transform_matrix"] is None:
                        inv_transform_matrix = np.eye(4, dtype=np.float64)[0:3]
                    else:
                        inv_transform_matrix = inverse_rigid_trans(sweep["transform_matrix"][0:3])
                    transform_matrix_list.append(inv_transform_matrix)
                    time_lag_list.append(times_sweep[0, 0])

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)
                sweep_idx = np.concatenate(sweep_idx_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                # res["lidar"]["combined"] = np.hstack([points, times, sweep_idx])
                res["lidar"]["combined"] = np.hstack([points, times])
                res["lidar"]["transform_matrix_list"] = transform_matrix_list
                res["lidar"]["time_lag_list"] = time_lag_list
                res["lidar"]["sweep_idxs"] = sweep_idx_list2  # include current frame

        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points
            if info.get('global_from_car', None) is not None:
                res["lidar"]["transform_matrix"] = info['global_from_car']

            if nsweeps > 1:
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],  # class type
                "obj_ids": info.get("obj_names", None),  # object track name/ID
            }
        else:
            pass

        return res, info
