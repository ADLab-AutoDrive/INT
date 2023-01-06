import os.path as osp

import torch
import torch.distributed as dist

from ...utils import master_only
from .base import LoggerHook


class TensorboardLoggerHook(LoggerHook):
    def __init__(self, log_dir=None, interval=10, ignore_last=True, reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.log_dir = log_dir

    def _get_max_memory(self, trainer):
        mem = torch.cuda.max_memory_allocated()
        mem_mb = torch.tensor(
            [mem / (1024 * 1024)], dtype=torch.int, device=torch.device("cuda")
        )
        if trainer.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    @master_only
    def before_run(self, trainer):
        if torch.__version__ >= "1.1":
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                )
        else:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError(
                    "Please install tensorboardX to use " "TensorboardLoggerHook."
                )

        if self.log_dir is None:
            self.log_dir = osp.join(trainer.work_dir, "tf_logs")
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def _get_info(self, trainer):
        log_dict = {}
        # Training mode if the output contains the key time
        log_dict["0-meta/epoch"] = trainer.epoch + 1
        log_dict["0-meta/iter"] = trainer.inner_iter + 1
        # Only record lr of the first param group
        log_dict["0-meta/lr"] = trainer.current_lr()[0]
        # statistic memory
        if torch.cuda.is_available():
            log_dict["0-meta/memory"] = self._get_max_memory(trainer)

        if trainer.world_size > 1:
            class_names = trainer.model.module.bbox_head.class_names
        else:
            class_names = trainer.model.bbox_head.class_names

        loc_elems_9 = ['cx', 'cy', 'cz', 'w', 'l', 'h', 'vx', 'vy', 'rsin', 'rcos']
        loc_elems_7 = ['cx', 'cy', 'cz', 'w', 'l', 'h', 'rsin', 'rcos']

        for idx, task_class_names in enumerate(class_names):
            pre_tag = '-'.join([str(idx+1)] + task_class_names)

            for out_key in ['loss', 'hm_loss', 'loc_loss', 'loc_loss_elem', 'num_positive']:
                if not out_key in trainer.log_buffer.output:
                    continue
                v = trainer.log_buffer.output[out_key]
                assert isinstance(v, list), type(v)
                if out_key == 'loc_loss_elem':
                    loc_elems = loc_elems_9 if len(v[idx]) == 10 else loc_elems_7
                    for i, vv in enumerate(v[idx]):
                        tag = f'{pre_tag}/{out_key}/{loc_elems[i]}'
                        log_dict[tag] = vv
                else:
                    tag = f'{pre_tag}/{out_key}'
                    log_dict[tag] = v[idx]

        return log_dict

    @master_only
    def log(self, trainer):
        log_dict = self._get_info(trainer)

        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, trainer.iter)

    # @master_only
    # def log(self, trainer):
    #     for var in trainer.log_buffer.output:
    #         if var in ["time", "data_time"]:
    #             continue
    #         tag = "{}/{}".format(var, trainer.mode)
    #         record = trainer.log_buffer.output[var]
    #         if isinstance(record, str):
    #             self.writer.add_text(tag, record, trainer.iter)
    #         else:
    #             self.writer.add_scalar(
    #                 tag, trainer.log_buffer.output[var], trainer.iter
    #             )

    @master_only
    def after_run(self, trainer):
        self.writer.close()
