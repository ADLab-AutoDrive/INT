import argparse
import json
import os
import sys

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import numpy as np
import torch
import yaml
from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument("--load_from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        print('no LOCAL RANK in os')
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()
    if args.resume_from == 'None':
        args.resume_from = None
    if args.load_from == 'None':
        args.load_from = None

    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

    cfg = Config.fromfile(args.config)
    print('args.local_rank:{}'.format(args.local_rank))
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    checkpoint_dir = cfg.checkpoint_config.get('out_dir', None)
    if checkpoint_dir is None:
        checkpoint_dir = cfg.work_dir
    try:
        pth_files = sorted([name for name in os.listdir(checkpoint_dir) if name.endswith('pth')])
    except Exception as e:
        print(e)
        pth_files = []
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if len(pth_files) > 0:  # higher priority, support online continue training!
        pth_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        cfg.resume_from = os.path.join(checkpoint_dir, pth_files[-1])

    if cfg.resume_from is not None and cfg.local_rank == 0:
        print(f"resume from {cfg.resume_from}")

    if args.load_from is not None:
        cfg.load_from = args.load_from

    if (cfg.resume_from is None) and (cfg.load_from is None):
        if 'freeze_pfn' in cfg.model:
            assert cfg.model.freeze_pfn is False, "when no pretrained model, should not freeze!"
        if 'freeze_backbone' in cfg.model:
            assert cfg.model.freeze_backbone is False, "when no pretrained model, should not freeze!"

    distributed = False
    if "WORLD_SIZE" in os.environ:
        print("WORLD_SIZE: {}", int(os.environ["WORLD_SIZE"]))
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()

    if args.autoscale_lr:
        cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    print("Distributed training: {}".format(distributed))
    print(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    # logger.info("Distributed training: {}".format(distributed))
    # logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    if args.local_rank == 0:
        # copy important files to backup
        backup_dir = os.path.join(cfg.work_dir, "det3d")
        os.makedirs(backup_dir, exist_ok=True)
        # os.system("cp -r * %s/" % backup_dir)
        # logger.info(f"Backup source files to {cfg.work_dir}/det3d")

    # set random seeds
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    cfg.data.train['distributed'] = distributed
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    if cfg.checkpoint_config is not None:
        # save det3d version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.text, CLASSES=datasets[0].CLASSES
        )

    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger,
        use_seq_sampler=True,
    )


if __name__ == "__main__":
    main()
