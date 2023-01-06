import os
import logging
import os.path as osp
import queue
import sys
import threading
import time
from collections import OrderedDict
import numpy as np
from functools import reduce

import torch
from torch.nn import functional as F
from det3d import torchie

from . import hooks
from .checkpoint import load_checkpoint, save_checkpoint
from .hooks import (
    CheckpointHook,
    Hook,
    IterTimerHook,
    LrUpdaterHook,
    OptimizerHook,
    EMAUpdateHook,
    lr_updater,
)
from .log_buffer import LogBuffer
from .priority import get_priority
from .utils import (
    all_gather,
    get_dist_info,
    get_host_info,
    get_time_str,
    obj_from_dict,
    synchronize,
)
from det3d.core.utils.transformations import inverse_rigid_trans, transform_xyz, rotz, transform_from_rot_trans
import warnings


def example_to_device(example, device, non_blocking=False) -> dict:
    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm",
                 "anno_box", "ind", "mask", 'cat'] or \
                k in ['voxels_list', 'num_points_list', 'num_voxels_list', 'coords_list', 'transform_matrix_list']:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "points",
            "num_voxels",
            "cyv_voxels",
            "cyv_num_voxels",
            "cyv_coordinates",
            "cyv_num_points",
            "gt_boxes_and_cls",
            'points_seg_label',
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = v1.to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def parse_second_losses(losses):
    log_vars = OrderedDict()
    loss = sum(losses["loss"])
    for loss_name, loss_value in losses.items():
        if loss_name == "loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]

    return loss, log_vars


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class Prefetcher(object):
    def __init__(self, dataloader):
        self.loader = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = example_to_device(
                self.next_input, torch.cuda.current_device(), non_blocking=False
            )

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input


class Trainer(object):
    """ A training helper for PyTorch

    Args:
        model:
        batch_processor:
        optimizer:
        workdir:
        log_level:
        logger:
    """

    def __init__(
            self,
            model,
            batch_processor,
            optimizer=None,
            lr_scheduler=None,
            work_dir=None,
            log_level=logging.INFO,
            logger=None,
            **kwargs,
    ):
        assert callable(batch_processor)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.batch_processor = batch_processor

        # Create work_dir
        if torchie.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            torchie.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError("'work_dir' must be a str or None")

        # Get model name from the model class
        if hasattr(self.model, "module"):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def init_optimizer(self, optimizer):
        """Init the optimizer

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`)

        Returns:
            :obj:`~torch.optim.Optimizer`

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD`>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(
                optimizer, torch.optim, dict(params=self.model.parameters())
            )
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer must be either an Optimizer object or a dict, "
                "but got {}".format(type(optimizer))
            )
        return optimizer

    def _add_file_handler(self, logger, filename=None, mode="w", level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, log_dir=None, level=logging.INFO):
        """Init the logger.

        Args:

        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - % (message)s", level=level
        )
        logger = logging.getLogger(__name__)
        if log_dir and self.rank == 0:
            filename = "{}.log".format(self.timestamp)
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def current_lr(self):
        if self.optimizer is None:
            raise RuntimeError("lr is not applicable because optimizer does not exist.")
        return [group["lr"] for group in self.optimizer.param_groups]

    def register_hook(self, hook, priority="NORMAL"):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`)
            priority (int or str or :obj:`Priority`)
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # Insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError(
                "'args' must be either a Hook object"
                " or dict, not {}".format(type(args))
            )

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location="cpu", strict=False):
        self.logger.info("load checkpoint from %s", filename)
        return load_checkpoint(self.model, filename, map_location, strict, self.logger)

    def save_checkpoint(
            self, out_dir, filename_tmpl="epoch_{}.pth", save_optimizer=True, meta=None
    ):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        # linkpath = osp.join(out_dir, "latest.pth")
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta,
                        exclude_keys=["infinite_feat_map", "infinite_heat_map", "infinite_pc"])
        # Use relative symlink
        # torchie.symlink(filename, linkpath)

    def batch_processor_inline(self, model, data, train_mode, **kwargs):

        if "local_rank" in kwargs:
            device = torch.device(kwargs["local_rank"])
        else:
            device = None

        # data = example_convert_to_torch(data, device=device)
        example = example_to_device(
            data, torch.cuda.current_device(), non_blocking=False
        )

        self.call_hook("after_data_to_device")

        if train_mode:
            losses = model(example, return_loss=True)
            self.call_hook("after_forward")
            loss, log_vars = parse_second_losses(losses)
            del losses

            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=-1  # TODO: FIX THIS
            )
            self.call_hook("after_parse_loss")

            return outputs
        else:
            return model(example, return_loss=False)

    def batch_processor_inline_infinite(self, model, data, train_mode, **kwargs):

        # data = example_convert_to_torch(data, device=device)
        example = example_to_device(
            data, torch.cuda.current_device(), non_blocking=False
        )

        if train_mode:
            self.call_hook("after_data_to_device")
            losses = model(example, return_loss=True, INFINITE_MODEL=True)
            self.call_hook("after_forward")
            loss, log_vars = parse_second_losses(losses)
            del losses

            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=-1  # TODO: FIX THIS
            )
            self.call_hook("after_parse_loss")

            return outputs
        else:
            return model(example, return_loss=False)

    def train(self, data_loader, epoch, **kwargs):

        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self.length = len(data_loader)
        self._max_iters = self._max_epochs * self.length
        self.call_hook("before_train_epoch")

        base_step = epoch * self.length

        # prefetcher = Prefetcher(data_loader)
        # for data_batch in BackgroundGenerator(data_loader, max_prefetch=3):
        for i, data_batch in enumerate(data_loader):
            global_step = base_step + i
            if self.lr_scheduler is not None:
                # print(global_step)
                self.lr_scheduler.step(global_step)

            self._inner_iter = i

            self.call_hook("before_train_iter")

            # outputs = self.batch_processor(self.model,
            #                                data_batch,
            #                                train_mode=True,
            #                                **kwargs)
            outputs = self.batch_processor_inline(
                self.model, data_batch, train_mode=True, **kwargs
            )

            if not isinstance(outputs, dict):
                raise TypeError("batch_processor() must return a dict")
            if "log_vars" in outputs:
                self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
            self.outputs = outputs
            self.call_hook("after_train_iter")
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1

    @staticmethod
    def inverse_rigid_trans(Tr):
        """Inverse a rigid body transform matrix (Bx4x4 as [R|t])
            [R'|-R't; 0|1]
        """
        inv_Tr = torch.zeros_like(Tr)  # Bx4x4
        inv_Tr[:, 0:3, 0:3] = Tr[:, 0:3, 0:3].transpose(1, 2)
        inv_Tr[:, 0:3, 3:4] = torch.bmm(-inv_Tr[:, 0:3, 0:3], Tr[:, 0:3, 3:4])
        inv_Tr[:, 3, 3] = 1
        return inv_Tr

    @staticmethod
    def get_transform_matrix(past_transform_matrix, cur_transform_matrix, metadata):
        def get_flip_matrix(flip_xy):
            flip_x_matrix, flip_y_matrix = torch.eye(4), torch.eye(4)
            flip_x_matrix[1, 1] = -1
            flip_y_matrix[0, 0] = -1

            flip_m = torch.eye(4)
            if flip_xy[0] is True:
                flip_m = torch.mm(flip_m, flip_x_matrix)
            if flip_xy[1] is True:
                flip_m = torch.mm(flip_m, flip_y_matrix)
            return flip_m

        use_aug = ('flip_xy' in metadata[0]) or ('rotation_ang' in metadata[0])
        if use_aug:
            flip_m_list = []
            rotation_m_list = []
            scale_m_list = []
            trans_m_list = []
            scale_m_inv_list = []
            for m in metadata:
                flip_m_list.append(get_flip_matrix(m['flip_xy']))

                r_m = transform_from_rot_trans(rotz(m['rotation_ang']), np.zeros(3))
                rotation_m_list.append(torch.from_numpy(r_m))

                s_m = np.eye(4) * m['noise_scale']
                s_m[3, 3] = 1
                scale_m_list.append(torch.from_numpy(s_m))

                t_m = np.eye(4)
                t_m[0:3, 3] = m['noise_trans'].reshape(-1)
                trans_m_list.append(torch.from_numpy(t_m))

                s_m_inv = np.eye(4) * 1. / m['noise_scale']
                s_m_inv[3, 3] = 1
                scale_m_inv_list.append(torch.from_numpy(s_m_inv))

            flip_m = torch.stack(flip_m_list).double()
            rotation_m = torch.stack(rotation_m_list).double()
            scale_m = torch.stack(scale_m_list).double()
            trans_m = torch.stack(trans_m_list).double()

            scale_m_inv = torch.stack(scale_m_inv_list).double()
            trans_m_inv = trans_m.clone()
            trans_m_inv[:, 0:3, 3] *= -1

            rotation_m_inv = Trainer.inverse_rigid_trans(rotation_m)
            cur_transform_matrix_inv = Trainer.inverse_rigid_trans(cur_transform_matrix)
            transform_matrix = reduce(torch.bmm,
                                      [trans_m_inv, scale_m_inv, rotation_m_inv, flip_m,
                                       cur_transform_matrix_inv, past_transform_matrix,
                                       flip_m, rotation_m, scale_m, trans_m])
            transform_matrix = Trainer.inverse_rigid_trans(transform_matrix)
            # print("use aug, transform matrix renewed!")
        else:
            past_transform_matrix_inverse = Trainer.inverse_rigid_trans(past_transform_matrix)
            transform_matrix = torch.bmm(past_transform_matrix_inverse, cur_transform_matrix)
        return transform_matrix  # 4x4

    @staticmethod
    def transform_featuremap(past_feat, transform_matrix, metadata, mode='nearest'):
        pc_range = metadata[0]['pc_range']
        xy_range = (pc_range[3] - pc_range[0], pc_range[4] - pc_range[1])

        transform_matrix = transform_matrix.to(past_feat.device)
        # transform_matrix = transform_matrix.to(torch.cuda.current_device(), non_blocking=False)
        transform_matrix = transform_matrix[:, 0:2, [0, 1, 3]]  # bs x 2 x 3
        transform_matrix[:, 0, 2] /= (0.5 * xy_range[0])
        transform_matrix[:, 1, 2] /= (0.5 * xy_range[1])
        grid = F.affine_grid(transform_matrix, past_feat.size()).float()
        affined_past_feat = F.grid_sample(past_feat, grid, mode=mode)
        return affined_past_feat

    @staticmethod
    def transform_point_cloud(past_pc, transform_matrix):
        '''
        past_pc: BxNx4
        transform_matrix: Bx4x4
        '''
        xyz, feat = past_pc[..., :3], past_pc[..., 3:]
        transform_matrix = Trainer.inverse_rigid_trans(transform_matrix)  # Bx4x4
        transform_matrix = transform_matrix.transpose(1, 2)[:, :, :3]  # Bx4x3
        transform_matrix = transform_matrix.to(past_pc.device).to(xyz.dtype)

        for bs_cnt in range(xyz.shape[0]):
            one_xyz, one_tm = xyz[bs_cnt], transform_matrix[bs_cnt]
            valid_mask = (one_xyz.sum(1) != 0)
            valid_xyz = one_xyz[valid_mask]
            one_padding = valid_xyz.new_ones(size=(valid_xyz.shape[0], 1))  # Nx1
            valid_xyz_hom = torch.cat([valid_xyz, one_padding], dim=-1)  # Nx4
            valid_xyz_trans = torch.mm(valid_xyz_hom, one_tm)
            xyz[bs_cnt][valid_mask] = valid_xyz_trans

        past_pc = torch.cat([xyz, feat], dim=-1)
        return past_pc

    def train_infinite(self, data_loader, epoch, **kwargs):

        # self.model.eval()
        net = self.model.module if hasattr(self.model, "module") else self.model
        if getattr(net, 'single_det', None) is not None:  # two stage
            net = net.single_det
        has_fm_fusion = getattr(net, 'fusion_method', None) is not None
        has_hm_fusion = getattr(net, 'hm_fusion_method', None) is not None
        has_pc_fusion = getattr(net, 'pc_fusion_method', None) is not None

        self.mode = "train"
        self.data_loader = data_loader
        self.length = len(data_loader)
        self._max_iters = self._max_epochs * self.length
        self.call_hook("before_train_epoch")

        base_step = epoch * self.length
        data_loader.sampler.epoch = epoch

        self.training_seq_len = None
        self.training_flag = None
        self._inner_iter = 0

        # prefetcher = Prefetcher(data_loader)
        # for data_batch in BackgroundGenerator(data_loader, max_prefetch=3):
        for i, data_batch in enumerate(data_loader):
            # if self.rank == 1:
            #     print(f"rank1, seq_tag: {data_batch['metadata'][0]['seq_tag']}")
            # print(f"seq_tag[0]: {data_batch['metadata'][0]['seq_tag']}")
            if self.training_seq_len is None:
                self.training_seq_len = data_loader.sampler.training_seq_len

            data_batch['training_seq'] = [i % self.training_seq_len + 1, self.training_seq_len]

            if self.training_flag is None:
                self.training_flag = data_loader.sampler.labelled_flag

            if (i % self.training_seq_len == 0):
                past_transform_matrix = data_batch["transform_matrix"].clone()
                # a new seq coming, clean history buffer
                if has_fm_fusion:
                    net.infinite_feat_map[...] = 0
                if has_hm_fusion:
                    net.infinite_heat_map[...] = 0
                if has_pc_fusion:
                    net.infinite_pc[...] = 0
                if self.rank == 0 and (has_hm_fusion or has_fm_fusion or has_pc_fusion):
                    self.logger.info(f"seq_iter: {i}, seq_tag[0]: {data_batch['metadata'][0]['seq_tag']} , "
                                     f"flushing history buffer!!!")
            else:
                cur_transform_matrix = data_batch["transform_matrix"].clone()

                if has_fm_fusion or has_hm_fusion or has_pc_fusion:
                    transform_matrix = self.get_transform_matrix(past_transform_matrix,
                                                                 cur_transform_matrix, data_batch['metadata'])

                if has_fm_fusion:
                    net.infinite_feat_map = self.transform_featuremap(net.infinite_feat_map, transform_matrix.clone(),
                                                                      data_batch['metadata'])
                if has_hm_fusion:
                    if epoch < (int(max(self.max_epochs / 10, 1))):
                        net.infinite_heat_map[...] = 0
                    else:
                        net.infinite_heat_map = \
                            self.transform_featuremap(net.infinite_heat_map, transform_matrix.clone(),
                                                      data_batch['metadata'], mode='bilinear')
                if has_pc_fusion:
                    net.infinite_pc = self.transform_point_cloud(net.infinite_pc, transform_matrix.clone())

                past_transform_matrix = cur_transform_matrix.clone()

            if self.training_flag[i] == 1:
                batch_size = len(data_batch['metadata'])
                with_gt_samples = sum([int(x['seq_tag'].split('::')[2]) for x in data_batch['metadata']])
                assert (with_gt_samples == batch_size)

                self.model.train()
                global_step = base_step + self._inner_iter
                if self.lr_scheduler is not None:
                    # print(global_step)
                    self.lr_scheduler.step(global_step)

                self._inner_iter += 1

                self.call_hook("before_train_iter")

                # outputs = self.batch_processor(self.model,
                #                                data_batch,
                #                                train_mode=True,
                #                                **kwargs)
                outputs = self.batch_processor_inline_infinite(
                    self.model, data_batch, train_mode=True, **kwargs
                )

                if not isinstance(outputs, dict):
                    raise TypeError("batch_processor() must return a dict")
                if "log_vars" in outputs:
                    self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
                self.outputs = outputs
                self.call_hook("after_train_iter")
                self._iter += 1
            else:
                self.model.eval()
                self.batch_processor_inline_infinite(
                    self.model, data_batch, train_mode=False,
                    **kwargs
                )

        self.call_hook("after_train_epoch")
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        self.call_hook("before_val_epoch")

        self.logger.info(f"work dir: {self.work_dir}")

        if self.rank == 0:
            prog_bar = torchie.ProgressBar(len(data_loader.dataset))

        detections = {}
        cpu_device = torch.device("cpu")

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook("before_val_iter")
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs
                )
            for output in outputs:
                token = output["metadata"]["token"]
                for k, v in output.items():
                    if k not in [
                        "metadata",
                    ]:
                        output[k] = v.to(cpu_device)
                detections.update(
                    {token: output, }
                )
                if self.rank == 0:
                    for _ in range(self.world_size):
                        prog_bar.update()

        synchronize()

        all_predictions = all_gather(detections)

        if self.rank != 0:
            return

        predictions = {}
        for p in all_predictions:
            predictions.update(p)

        # torch.save(predictions, "final_predictions_debug.pkl")
        # TODO fix evaluation module
        result_dict, _ = self.data_loader.dataset.evaluation(
            predictions, output_dir=self.work_dir
        )

        self.logger.info("\n")
        for k, v in result_dict["results"].items():
            self.logger.info(f"Evaluation {k}: {v}")

        self.call_hook("after_val_epoch")

    def resume(self, checkpoint, resume_optimizer=True, map_location="default"):
        if map_location == "default":
            checkpoint = self.load_checkpoint(
                checkpoint, map_location='cuda:{}'.format(torch.cuda.current_device())  # TODO: FIX THIS!!
            )
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)

        self._epoch = checkpoint["meta"]["epoch"]
        self._iter = checkpoint["meta"]["iter"]
        if "optimizer" in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("resumed epoch %d, iter %d", self.epoch, self.iter)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """ Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`])
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs.
            max_epochs (int)
        """
        assert isinstance(data_loaders, list)
        assert torchie.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info(
            "Start running, host: %s, work_dir: %s", get_host_info(), work_dir
        )
        self.logger.info("workflow: %s, max: %d epochs", workflow, max_epochs)
        self.call_hook("before_run")

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError(
                            "Trainer has no method named '{}' to run an epoch".format(
                                mode
                            )
                        )
                    epoch_runner = getattr(self, mode)
                elif callable(mode):
                    epoch_runner = mode
                else:
                    raise TypeError(
                        "mode in workflow must be a str or "
                        "callable function not '{}'".format(type(mode))
                    )

                for _ in range(epochs):
                    if mode == "train" and self.epoch >= max_epochs:
                        return
                    elif mode == "val":
                        epoch_runner(data_loaders[i], **kwargs)
                    else:
                        epoch_runner(data_loaders[i], self.epoch, **kwargs)

        # time.sleep(1)
        self.call_hook("after_run")

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert "policy" in lr_config
            hook_name = lr_config["policy"].title() + "LrUpdaterHook"
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError(
                "'lr_config' must be eigher a LrUpdaterHook object"
                " or dict, not '{}'".format(type(lr_config))
            )

    def register_logger_hooks(self, log_config):
        log_interval = log_config["interval"]
        for info in log_config["hooks"]:
            logger_hook = obj_from_dict(
                info, hooks, default_args=dict(interval=log_interval)
            )
            self.register_hook(logger_hook, priority="VERY_LOW")

    def register_training_hooks(
            self, lr_config, optimizer_config=None, checkpoint_config=None, log_config=None
    ):
        """Register default hooks for training.

        Default hooks include:
            - LrUpdaterHook
            - OptimizerStepperHook
            - CheckpointSaverHook
            - IterTimerHook
            - LoggerHook(s)
        """
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        if lr_config is not None:
            assert self.lr_scheduler is None
            self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)
