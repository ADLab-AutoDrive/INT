import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle
import time

from det3d.torchie.trainer.trainer import example_to_device, Trainer


def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--work_dir", required=False, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")
    parser.add_argument(
        "--test_frames",
        type=int,
        default=None,
        help="number of frames to use.",
    )
    parser.add_argument(
        "--sampled_interval",
        type=int,
        default=None,
    )

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def batch_processor_test(model, data):
    # data = example_convert_to_torch(data, device=device)
    example = example_to_device(
        data, torch.cuda.current_device(), non_blocking=False
    )
    return model(example, return_loss=False)


def main():
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        if args.sampled_interval is not None:
            cfg.data.val['sampled_interval'] = args.sampled_interval
        dataset = build_dataset(cfg.data.val)

    if args.test_frames is not None:
        cfg.test_seq_len = args.test_frames

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu if not args.speed_test else 0,
        dist=distributed,
        shuffle=False,
        use_seq_sampler=True,
        max_epoches=cfg.total_epochs,
        max_training_seq=cfg.max_training_seq,
        test_seq_len=cfg.test_seq_len,
        repeat_test=cfg.get("repeat_test", True),
    )

    print('load {}'.format(args.checkpoint))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            # find_unused_parameters=True,
            find_unused_parameters=False,
        )
    else:
        # model = fuse_bn_recursively(model)
        model = model.cuda()

    model.eval()
    mode = "val"

    logger.info(f"work dir: {cfg.work_dir}")
    if cfg.local_rank == 0:
        # prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)
        prog_bar = torchie.ProgressBar(len(data_loader.sampler))

    detections = {}
    cpu_device = torch.device("cpu")

    start = int(len(data_loader) / 3)
    end = int(len(data_loader) * 2 / 3)

    time_start = 0
    time_end = 0

    net = model.module if hasattr(model, "module") else model
    if getattr(net, 'single_det', None) is not None:  # two stage
        net = net.single_det
    has_fm_fusion = getattr(net, 'fusion_method', None) is not None
    has_hm_fusion = getattr(net, 'hm_fusion_method', None) is not None
    has_pc_fusion = getattr(net, 'pc_fusion_method', None) is not None
    test_seq_len = None
    labelled_flag = None

    for i, data_batch in enumerate(data_loader):
        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        # print(f"seq_tag[0]: {data_batch['metadata'][0]['seq_tag']}")
        if test_seq_len is None:
            test_seq_len = data_loader.sampler.test_seq_len

        data_batch['training_seq'] = [i % test_seq_len + 1, test_seq_len]

        if labelled_flag is None:
            labelled_flag = data_loader.sampler.labelled_flag

        if (i % test_seq_len == 0):
            past_transform_matrix = data_batch["transform_matrix"].clone()
            # a new seq coming, clean history buffer
            if has_fm_fusion:
                net.infinite_feat_map[...] = 0
            if has_hm_fusion:
                net.infinite_heat_map[...] = 0
            if has_pc_fusion:
                net.infinite_pc[...] = 0
            if args.local_rank == 0:
                logger.info(f"seq_iter: {i}, seq_tag[0]: {data_batch['metadata'][0]['seq_tag']} , "
                            f"flushing featmap buffer!!!")
        else:
            cur_transform_matrix = data_batch["transform_matrix"].clone()

            if has_fm_fusion or has_hm_fusion or has_pc_fusion:
                transform_matrix = Trainer.get_transform_matrix(past_transform_matrix,
                                                                cur_transform_matrix, data_batch['metadata'])

            if has_fm_fusion:
                net.infinite_feat_map = Trainer.transform_featuremap(net.infinite_feat_map, transform_matrix.clone(),
                                                                     data_batch['metadata'])
            if has_hm_fusion:
                net.infinite_heat_map = Trainer.transform_featuremap(net.infinite_heat_map, transform_matrix.clone(),
                                                                     data_batch['metadata'], mode='bilinear')
            if has_pc_fusion:
                net.infinite_pc = Trainer.transform_point_cloud(net.infinite_pc, transform_matrix.clone())

            past_transform_matrix = cur_transform_matrix.clone()

        with torch.no_grad():
            # outputs = batch_processor(
            #     model, data_batch, train_mode=False, local_rank=args.local_rank,
            # )
            if labelled_flag[i] == 0:
                batch_processor_test(model, data_batch)
                continue

            outputs = batch_processor_test(model, data_batch)

        for output in outputs:
            token = output["metadata"]["token"]
            assert token is not None
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output, }
            )
            if args.local_rank == 0:
                prog_bar.update()

    synchronize()

    all_predictions = all_gather(detections)

    print("\n Total time per frame: ", (time_end - time_start) / (end - start))

    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)

    save_pred(predictions, cfg.work_dir)

    # output_dir = os.path.join(cfg.work_dir, f'test_seq_len_{cfg.test_seq_len}')
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    output_dir = cfg.work_dir
    if args.testset:
        result_dict, _ = dataset.evaluation_pd(copy.deepcopy(predictions), output_dir=output_dir)
    else:
        result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=output_dir, testset=args.testset)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"


if __name__ == "__main__":
    main()
