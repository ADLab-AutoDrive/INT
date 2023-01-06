import copy
import pickle
from pathlib import Path
import os
import numpy as np

from det3d.core import box_np_ops
from det3d.datasets.dataset_factory import get_dataset
from tqdm import tqdm
from collections import defaultdict

dataset_name_map = {
    "NUSC": "NuScenesDataset",
    "WAYMO": "WaymoDataset"
}


def create_groundtruth_database(
        dataset_class_name,
        data_path,
        info_path=None,
        used_classes=None,
        db_path=None,
        dbinfo_path=None,
        relative_path=True,
        output_dir=None,
        **kwargs,
):
    pipeline = [
        {
            "type": "LoadPointCloudFromFile",
            "dataset": dataset_name_map[dataset_class_name],
        },
        {"type": "LoadPointCloudAnnotations", "with_bbox": True},
    ]

    if "nsweeps" in kwargs:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path,
            root_path=data_path,
            pipeline=pipeline,
            test_mode=True,
            nsweeps=kwargs["nsweeps"],
        )
        nsweeps = dataset.nsweeps
    else:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path, root_path=data_path, test_mode=True, pipeline=pipeline
        )
        nsweeps = 1

    root_path = Path(data_path)

    if output_dir is None:
        output_dir = root_path
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_class_name in ["WAYMO", "NUSC"]:
        if db_path is None:
            db_path = output_dir / "gt_database_{:02d}sweeps_withvelo".format(nsweeps)
        if dbinfo_path is None:
            dbinfo_path = output_dir / "dbinfos_train_{:02d}sweeps_withvelo.pkl".format(nsweeps)
    else:
        raise NotImplementedError()

    if dataset_class_name == "NUSC":
        point_features = 5
    elif dataset_class_name == "WAYMO":
        point_features = 5 if nsweeps == 1 else 6
    else:
        raise NotImplementedError()

    db_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    group_counter = 0

    for index in tqdm(range(len(dataset))):
        image_idx = index
        # modified to nuscenes
        sensor_data = dataset.get_sensor_data(index)
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]

        if nsweeps > 1 or dataset_class_name == 'NUSC':
            points = sensor_data["lidar"]["combined"]
        else:
            points = sensor_data["lidar"]["points"]

        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]

        if dataset_class_name == 'WAYMO':
            # waymo dataset contains millions of objects and it is not possible to store
            # all of them into a single folder
            # we randomly sample a few objects for gt augmentation
            # We keep all cyclist as they are rare 
            if index % 4 != 0:
                mask = (names == 'VEHICLE')
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]

            if index % 2 != 0:
                mask = (names == 'PEDESTRIAN')
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]

        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            continue
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            if (used_classes is None) or names[i] in used_classes:
                filename = f"{image_idx}_{names[i]}_{i}.bin"
                dirpath = os.path.join(str(db_path), names[i])
                os.makedirs(dirpath, exist_ok=True)

                filepath = os.path.join(str(db_path), names[i], filename)
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "w") as f:
                    try:
                        gt_points[:, :point_features].tofile(f)
                    except:
                        print("process {} files".format(index))
                        break

            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_dump_path = os.path.join(db_path.stem, names[i], filename)
                else:
                    db_dump_path = str(filepath)

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    print("dataset length: ", len(dataset))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, "wb") as f:
        pickle.dump(all_db_infos, f)


def create_groundtruth_database_seq_wise(
        dataset_class_name,
        data_path,
        info_path=None,
        used_classes=None,
        db_path=None,
        dbinfo_path=None,
        relative_path=True,
        output_dir=None,
        seq_num=None,
        **kwargs,
):
    pipeline = [
        {
            "type": "LoadPointCloudFromFile",
            "dataset": dataset_name_map[dataset_class_name],
        },
        {"type": "LoadPointCloudAnnotations", "with_bbox": True},
    ]

    dataset = get_dataset(dataset_class_name)(
        info_path=info_path, root_path=data_path, nsweeps=1, test_mode=True, pipeline=pipeline,
        use_seq_info=True,
    )

    root_path = Path(data_path)

    if output_dir is None:
        output_dir = root_path
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_class_name in ["WAYMO", "NUSC"]:
        if db_path is None:
            db_path = output_dir / "gt_database_seq_withvelo"
        if dbinfo_path is None:
            dbinfo_path = output_dir / "dbinfos_train_seq_withvelo.pkl"
    else:
        raise NotImplementedError()

    if dataset_class_name == "NUSC":
        point_features = 4
    elif dataset_class_name == "WAYMO":
        point_features = 5
    else:
        raise NotImplementedError()

    db_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = defaultdict(list)
    obj_pc_dict = defaultdict(list)
    obj_info_dict = defaultdict(list)

    seq_name = None
    seq_cnt = 0
    for index in tqdm(range(len(dataset))):
        sensor_data = dataset.get_sensor_data(index)
        cur_seq_name = sensor_data['metadata']['seq_tag'].split('::')[0]
        if seq_name is None:
            seq_name = cur_seq_name
        if (cur_seq_name != seq_name) or (index == (len(dataset) - 1)):
            seq_cnt += 1
            for obj_id in obj_info_dict:
                obj_info_list = obj_info_dict[obj_id]
                obj_name = obj_info_list[0]['name']

                # waymo dataset contains millions of objects and it is not possible to store
                # all of them into a single folder
                # we randomly sample a few objects for gt augmentation
                # We keep all cyclist as they are rare
                if (dataset_class_name == 'WAYMO') and (obj_name == 'VEHICLE') and (np.random.rand() < 0.75):
                    continue
                if (dataset_class_name == 'WAYMO') and (obj_name == 'PEDESTRIAN') and (np.random.rand() < 0.5):
                    continue

                # save pointcloud
                dirpath = os.path.join(str(db_path), obj_name)
                os.makedirs(dirpath, exist_ok=True)
                obj_pc_list = obj_pc_dict[obj_id]
                for i, (obj_pc, obj_info) in enumerate(zip(obj_pc_list, obj_info_list)):
                    filename = f'{seq_name}_{obj_id}_{i}.bin'
                    rel_save_path = os.path.join(db_path.stem, obj_name, filename)
                    abs_save_path = os.path.join(output_dir, rel_save_path)
                    with open(abs_save_path, "w") as f:
                        obj_pc[:, :point_features].tofile(f)

                    obj_info['path'] = rel_save_path

                # update db info
                all_db_infos[obj_name].append(copy.deepcopy(obj_info_list))

            # clear cur_db_info_dict
            obj_pc_dict = defaultdict(list)
            obj_info_dict = defaultdict(list)

            seq_name = cur_seq_name
            if (seq_num is not None) and (seq_cnt == seq_num):
                break

        points = sensor_data["lidar"]["points"]

        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]
        ids = annos["obj_ids"]

        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            continue
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            if (used_classes is None) or names[i] in used_classes:
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]

                db_info = {
                    "name": names[i],
                    "path": None,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }

                obj_id = ids[i]
                obj_pc_dict[obj_id].append(gt_points)
                obj_info_dict[obj_id].append(db_info)

    print("dataset length: ", len(dataset))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, "wb") as f:
        pickle.dump(all_db_infos, f)
