import copy
from pathlib import Path
import pickle

import fire, os

from det3d.datasets.utils.create_gt_database import create_groundtruth_database, create_groundtruth_database_seq_wise
from det3d.datasets.waymo import waymo_common as waymo_ds


def waymo_data_prep(root_path, split, nsweeps=1, output_dir=None):
    output_dir = waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps, output_dir=output_dir)
    if split == 'train':
        create_groundtruth_database(
            "WAYMO",
            root_path,
            output_dir / "infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps,
            output_dir=output_dir,
        )


def waymo_data_prep_seqwise(root_path, split, nsweeps=1, output_dir=None):
    waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps, output_dir=output_dir, make_seq_wise_info=True)
    if split == 'train':
        if output_dir is None:
            output_dir = root_path
        else:
            output_dir = Path(output_dir)
        create_groundtruth_database_seq_wise(
            "WAYMO",
            root_path,
            output_dir / "infos_train_seq_filter_zero_gt.pkl".format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps,
            output_dir=output_dir,
            seq_num=None,
        )


if __name__ == "__main__":
    # fire.Fire()
    # waymo_data_prep(root_path="./data/Waymo", split='train')
    waymo_data_prep_seqwise(root_path="./data/Waymo", split='test')
