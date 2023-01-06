import platform
from functools import partial

from det3d.torchie.parallel import collate, collate_kitti
from det3d.torchie.trainer import get_dist_info
from torch.utils.data import DataLoader

from .sampler import (
    DistributedGroupSampler,
    DistributedSampler,
    DistributedSamplerV2,
    GroupSampler,
    DistributedSeqSampler,
    SeqSampler,
)

if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(
        dataset, batch_size, workers_per_gpu, num_gpus=1, dist=True, **kwargs
):
    print('in build_dataloader, dist: {}'.format(dist))

    shuffle = kwargs.get("shuffle", True)

    use_seq_sampler = kwargs.get("use_seq_sampler", False)
    if use_seq_sampler:
        max_training_seq = kwargs["max_training_seq"]
        training_seq_len = kwargs.get("training_seq_len", None)
        if training_seq_len == 'None':
            training_seq_len = None
        max_epoches = kwargs["max_epoches"]

        test_seq_len = kwargs.get("test_seq_len", None)
        repeat_test = kwargs.get("repeat_test", True)

    if dist:
        rank, world_size = get_dist_info()
        # sampler = DistributedSamplerV2(dataset,
        #                      num_replicas=world_size,
        #                      rank=rank,
        #                      shuffle=shuffle)

        if use_seq_sampler:
            sampler = DistributedSeqSampler(dataset, max_training_seq, batch_size, max_epoches, shuffle,
                                            world_size, rank, test_seq_len=test_seq_len,
                                            training_seq_len=training_seq_len, repeat_test=repeat_test)
        else:
            if shuffle:
                sampler = DistributedGroupSampler(dataset, batch_size, world_size, rank)
            else:
                sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
        batch_size = batch_size
        num_workers = workers_per_gpu
    else:
        if use_seq_sampler:
            sampler = SeqSampler(dataset, max_training_seq, batch_size, max_epoches, shuffle=shuffle,
                                 test_seq_len=test_seq_len, training_seq_len=training_seq_len, repeat_test=repeat_test)
        else:
            sampler = GroupSampler(dataset, batch_size) if shuffle else None
            # sampler = None
        batch_size = num_gpus * batch_size
        num_workers = num_gpus * workers_per_gpu

    # TODO change pin_memory
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=collate_kitti,
        # pin_memory=True,
        pin_memory=False,
    )

    return data_loader
