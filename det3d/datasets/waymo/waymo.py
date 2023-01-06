import sys
import os
import copy
import pickle
import json
import random
import operator
from numba.cuda.simulator.api import detect
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

from det3d.datasets.custom import PointCloudDataset

from det3d.datasets.registry import DATASETS
from det3d.torchie.trainer import get_dist_info
import warnings


@DATASETS.register_module
class WaymoDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, elongation

    def __init__(
            self,
            info_path,
            root_path,
            cfg=None,
            pipeline=None,
            class_names=None,
            test_mode=False,
            nsweeps=1,
            **kwargs,
    ):
        super(WaymoDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self.ema_exp = kwargs.get('ema_exp', False)
        self.pc_range = kwargs.get('pc_range', None)
        self.sampled_interval = kwargs.get('sampled_interval', 1)
        print(f"self.sampled_interval: {self.sampled_interval}")
        # print(kwargs)
        self.use_seq_info = kwargs.get('use_seq_info', False)

        self.nsweeps = nsweeps
        print("Using {} sweeps".format(nsweeps))

        self._info_path = info_path
        self._class_names = class_names

        self._waymo_infos = self.load_infos()

        if self.use_seq_info:
            self.seq_tag = [x["seq_tag"] for x in self._waymo_infos]
            self.label_interval = 1

        self._num_point_features = WaymoDataset.NumPointFeatures
        if self.nsweeps > 1 and not self.ema_exp:
            self._num_point_features += 1  # add timestamp dimension

        self._set_group_flag()

        self.distributed = kwargs.get('distributed', False)
        if self.distributed:
            self.rank, world_size = get_dist_info()

    def reset(self):
        assert False

    def load_infos(self):
        with open(self._info_path, "rb") as f:
            _waymo_infos_all = pickle.load(f)

        if self.sampled_interval > 1:
            if self.use_seq_info:
                sampled_waymo_infos = {}
                keys = list(_waymo_infos_all.keys())
                for key in keys[::self.sampled_interval]:
                    sampled_waymo_infos[key] = _waymo_infos_all[key]
                _waymo_infos_all = sampled_waymo_infos
            else:
                sampled_waymo_infos = []
                for k in range(0, len(_waymo_infos_all), self.sampled_interval):
                    sampled_waymo_infos.append(_waymo_infos_all[k])
                _waymo_infos_all = sampled_waymo_infos
            print(f'sample 1/{self.sampled_interval} data for use!')

        if isinstance(_waymo_infos_all, dict):
            _waymo_infos = []
            for v in _waymo_infos_all.values():
                _waymo_infos.extend(v)
        else:
            _waymo_infos = _waymo_infos_all

        return _waymo_infos

    def __len__(self):
        # if not hasattr(self, "_waymo_infos"):
        #     self.load_infos(self._info_path)

        return len(self._waymo_infos)

    def get_sensor_data(self, idx):
        info = self._waymo_infos[idx]

        if self.use_seq_info and (not self.test_mode):
            if self.distributed:
                pkl_path = f'random_seeds_{self.rank}.pkl'
            else:
                pkl_path = 'random_seeds.pkl'

            with open(pkl_path, 'rb') as f:
                random_seeds = pickle.load(f)
            info['random_seed'] = random_seeds[idx]['seed']
            info['train_seq_len_and_cur_frame'] = random_seeds[idx]['train_seq_len_and_cur_frame']

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "annotations": None,
                "nsweeps": self.nsweeps,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "info_prefix": Path(self._info_path).parent,  # info prefix
                "num_point_features": self._num_point_features,
                "token": info["token"],
                "seq_tag": info.get("seq_tag", None),
                "pc_range": self.pc_range,
                "use_seq_info": self.use_seq_info,
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "type": "WaymoDataset",
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation_pd(self, detections, output_dir=None, testset=False):
        from .waymo_common import _create_pd_detection, reorganize_info

        infos = self._waymo_infos
        infos = reorganize_info(infos)

        _create_pd_detection(detections, infos, output_dir)

        print("use waymo devkit tool for evaluation")

        return None, None

    def evaluation(self, det_annos, output_dir=None, testset=False):
        if 'gt_boxes' not in self._waymo_infos[0].keys():
            print('No ground-truth boxes for evaluation')
            return None, None

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=self._class_names,
                distance_thresh=1000
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0].item()
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = []
        eval_gt_annos = []
        for gt_anno in self._waymo_infos:
            ori_det = det_annos[gt_anno['token']]
            det = {}
            box3d = ori_det['box3d_lidar'].detach().cpu().numpy()
            box3d[:, -1] = -box3d[:, -1] - np.pi / 2
            box3d = box3d[:, [0, 1, 2, 4, 3, 5, -1]]
            det['box3d_lidar'] = box3d
            det['scores'] = ori_det['scores'].detach().cpu().numpy()
            labels = ori_det["label_preds"].detach().cpu().numpy()
            det['name'] = np.array([self._class_names[label] for label in labels])
            eval_det_annos.append(det)

            gt = {}
            box3d = gt_anno['gt_boxes']
            box3d[:, -1] = -box3d[:, -1] - np.pi / 2
            box3d = box3d[:, [0, 1, 2, 4, 3, 5, -1]]
            gt['gt_box3d_lidar'] = box3d
            gt['name'] = gt_anno['gt_names']
            gt['num_points_in_gt'] = gt_anno['num_points_in_gt']
            gt['difficulty'] = gt_anno['difficulty']
            eval_gt_annos.append(gt)

        ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)

        l2_vel = ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APH']
        l2_ped = ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APH']
        l2_cyc = ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APH']
        ap_dict['OBJECT_TYPE_ALL_LEVEL_2/MAPH'] = (l2_vel + l2_ped + l2_cyc) / 3.0

        ap_result_str += f"OBJECT_TYPE_ALL_LEVEL_2/MAPH: {ap_dict['OBJECT_TYPE_ALL_LEVEL_2/MAPH']}\n"

        res = {
            "results": {"waymo": ap_result_str, },
            "detail": {},
        }
        with open(Path(output_dir) / 'metrics_summary.json', 'w') as f:
            json.dump(ap_dict, f)

        return res, ap_result_str
