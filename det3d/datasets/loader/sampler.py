from __future__ import division
import os
import math
import pickle

import numpy as np
import torch
import math
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

from det3d.torchie.trainer import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSamplerV2(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class GroupSampler(Sampler):
    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, "flag")
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += (
                    int(np.ceil(size / self.samples_per_gpu)) * self.samples_per_gpu
            )

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(
                np.ceil(size / self.samples_per_gpu)
            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, indice[:num_extra]])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu: (i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, samples_per_gpu=1, num_replicas=None, rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, "flag")
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += (
                    int(
                        math.ceil(
                            self.group_sizes[i]
                            * 1.0
                            / self.samples_per_gpu
                            / self.num_replicas
                        )
                    )
                    * self.samples_per_gpu
            )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size), generator=g))].tolist()
                extra = int(
                    math.ceil(size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                indice += indice[:extra]
                indices += indice

        assert len(indices) == self.total_size

        indices = [
            indices[j]
            for i in list(
                torch.randperm(len(indices) // self.samples_per_gpu, generator=g)
            )
            for j in range(i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class SeqSampler(Sampler):
    def __init__(self, dataset, max_training_seq=100, samples_per_gpu=1, max_epoches=20, shuffle=False,
                 test_seq_len=None, training_seq_len=None, repeat_test=True):
        assert hasattr(dataset, "seq_tag")
        assert hasattr(dataset, "label_interval")
        self.dataset = dataset
        self.test_mode = dataset.test_mode
        self.max_training_seq = max_training_seq
        self.batch_size = samples_per_gpu
        self.max_epochs = max_epoches
        self.shuffle = shuffle
        self.seq_tag = dataset.seq_tag  # list of "seq_name::frame_idx"
        self.label_interval = dataset.label_interval
        self.test_seq_len = test_seq_len
        self.repeat_test = repeat_test
        self.fixed_training_seq_len = training_seq_len

        self.seq_dict = {}
        for data_idx, seq_tag in enumerate(self.seq_tag):
            seq_name, frame_idx, with_gt = seq_tag.split('::')
            frame_idx, with_gt = int(frame_idx), int(with_gt)
            if not seq_name in self.seq_dict:
                self.seq_dict[seq_name] = []
            self.seq_dict[seq_name].append([frame_idx, data_idx, with_gt])

        labeled_samples = 0
        for seq_name, idxs in self.seq_dict.items():
            self.seq_dict[seq_name] = np.array(sorted(idxs, key=lambda x: x[0]))
            labeled_samples += (self.seq_dict[seq_name][:, 2] == 1).sum()
        self.labeled_samples = labeled_samples

        self.epoch = 0  # maybe rewrite outside, influence self.training_seq_len
        self.samples_per_batchID = int(
            np.ceil(self.labeled_samples / self.batch_size))  # total samples for one batch_id
        self.num_samples = self.samples_per_batchID * self.batch_size
        self.total_iter_samples = None

    def _re_indices(self, test=False):
        if not test:
            self.epoch += 1
        if self.fixed_training_seq_len is None:
            # self.training_seq_len = max(int(self.epoch / self.max_epochs * self.max_training_seq), 1)
            # if self.epoch >= int(0.8 * self.max_epochs):
            #     self.training_seq_len = self.max_training_seq
            # better way to set DTSL. <0.25 1, >0.75 max_l, (0.25~0.75) linear change.
            self.training_seq_len = max(
                int(min(1, max(0, 2.0 * self.epoch / self.max_epochs - 0.5)) * self.max_training_seq), 1)

        else:
            self.training_seq_len = self.fixed_training_seq_len
        print("start re-indices!")

        labelled_samples_per_training_seq = int(np.ceil(self.training_seq_len / self.label_interval))
        num_training_seq_per_batch = int(np.ceil(self.samples_per_batchID / labelled_samples_per_training_seq))

        labelled_flag_template = np.zeros((self.training_seq_len,), dtype=np.int32)
        labelled_flag_template[[-self.label_interval * i - 1 for i in range(labelled_samples_per_training_seq)]] = 1
        labelled_flag_template = list(labelled_flag_template)

        # split original seq to training_seq
        training_seq_list = []
        for seq_name, idxs in self.seq_dict.items():
            labelled_samples = np.where(idxs[:, 2] == 1)[0]
            if len(labelled_samples) < labelled_samples_per_training_seq:
                print(f"{seq_name}: {len(labelled_samples)} is not long enough for"
                      f" training_seq_len: {self.training_seq_len}")
                # continue
            num_training_seq = int(np.ceil(len(labelled_samples) / labelled_samples_per_training_seq))
            training_seq_end_idxs = [labelled_samples[labelled_samples_per_training_seq * i - 1] for i in
                                     range(1, num_training_seq)]
            training_seq_end_idxs += [labelled_samples[-1]]
            training_seq_se_idxs = [[max(-1, end - self.training_seq_len), end] for end in training_seq_end_idxs]

            # check last seq to avoid same idx in multiple training seq (so as to keep random seed unique!)
            if len(training_seq_se_idxs) > 1:
                if training_seq_se_idxs[-1][0] < training_seq_se_idxs[-2][1]:  # last start > last last end
                    training_seq_se_idxs[-1][0] = training_seq_se_idxs[-2][1]

            training_seq_data_idxs = [list(idxs[range(start + 1, end + 1), 1]) for start, end in training_seq_se_idxs]

            # complete the first and last seq
            if len(training_seq_data_idxs[0]) < self.training_seq_len:
                need_len = self.training_seq_len - len(training_seq_data_idxs[0])
                training_seq_data_idxs[0] = [training_seq_data_idxs[0][0]] * need_len + training_seq_data_idxs[0]
            if len(training_seq_data_idxs[-1]) < self.training_seq_len:
                need_len = self.training_seq_len - len(training_seq_data_idxs[-1])
                training_seq_data_idxs[-1] = [training_seq_data_idxs[-1][0]] * need_len + training_seq_data_idxs[-1]

            assert np.all(np.array([len(data_idx) == self.training_seq_len for data_idx in training_seq_data_idxs]))
            training_seq_list.extend(training_seq_data_idxs)

        if self.shuffle:
            np.random.shuffle(training_seq_list)

        if test and len(training_seq_list) > (num_training_seq_per_batch * self.batch_size):
            new_num_training_seq_per_batch = int(np.ceil(len(training_seq_list) / self.batch_size))
            print(f'In order to take every labelled samples to indices, enlarge num_training_seq_per_batch from'
                  f' {num_training_seq_per_batch} to {new_num_training_seq_per_batch}')
            num_training_seq_per_batch = new_num_training_seq_per_batch

        # use training_seq_list to assemble batch!
        ## check the last seq in a batch
        remainder = self.samples_per_batchID % labelled_samples_per_training_seq
        if remainder != 0:
            last_training_seq_slice = - self.label_interval * remainder
        self.labelled_flag = []
        for i in range(num_training_seq_per_batch):
            if (i == (num_training_seq_per_batch - 1)) and (remainder != 0) and (not test):
                self.labelled_flag.extend(labelled_flag_template[last_training_seq_slice:])
            else:
                self.labelled_flag.extend(labelled_flag_template)

        base_i = 0
        indices = []
        for bs in range(self.batch_size):
            indices_per_batch = []
            for i in range(num_training_seq_per_batch):
                new_i = base_i + i
                if new_i >= len(training_seq_list):
                    new_i = np.random.randint(len(training_seq_list))
                if (i == (num_training_seq_per_batch - 1)) and (remainder != 0) and (not test):
                    indices_per_batch.extend(training_seq_list[new_i][last_training_seq_slice:])
                else:
                    indices_per_batch.extend(training_seq_list[new_i])
            base_i += num_training_seq_per_batch
            indices.append(indices_per_batch)

        ## check! the labelled samples in resembled seqs should be equal(or larger) to samples_per_batchID
        labelled_flag_sum = sum(self.labelled_flag)
        if self.samples_per_batchID >= labelled_samples_per_training_seq:
            if test:
                assert labelled_flag_sum >= self.samples_per_batchID
            else:
                assert labelled_flag_sum == self.samples_per_batchID
        else:
            print(f"Batch size {self.batch_size} is too big!")
            raise InterruptedError
        # else:
        #     if labelled_flag_sum != self.samples_per_batchID:
        #         assert labelled_flag_sum > self.samples_per_batchID
        #         chosen_idx = np.random.choice(np.where(self.labelled_flag == 1)[0],
        #                                       labelled_flag_sum - self.samples_per_batchID, replace=False)
        #         tmp = np.array(self.labelled_flag)
        #         tmp[chosen_idx] = 0
        #         self.labelled_flag = list(tmp)

        if test:
            # check if every labelled sample is in indices
            check_set = set()
            for idx in indices:
                check_set.update(tuple(idx))
            assert len(check_set) == self.labeled_samples, \
                f'not all labelled samples in indices, {len(check_set)}<{self.labeled_samples}'

        indices = np.array(indices)  # batch_size * seq_len

        ## set random seed for same training_seq
        random_seeds = {}
        # set random seed for same training seq
        for one_batch_idxs in indices:
            for i, idx in enumerate(one_batch_idxs):
                if i % self.training_seq_len == 0:
                    seed = np.random.randint(1e8)
                if not idx in random_seeds:  # in case one sample in two training_seq, keep first seq random seed
                    random_seeds[idx] = dict(
                        seed=seed,
                        train_seq_len_and_cur_frame=[self.training_seq_len, i % self.training_seq_len]
                    )
        # with open(out_path, 'wb') as f:
        #     pickle.dump(random_seeds, f)

        indices = indices.transpose().reshape(-1)

        indices = indices.astype(np.int64).tolist()
        self.total_iter_samples = len(indices)

        print(f"ep: {self.epoch}, training_seq_len: {self.training_seq_len}, "
              f"labelled_samples_per_batchID: {self.samples_per_batchID}, "
              f"total_labelled_samples: {self.num_samples}, "
              f"num_training_seq_per_batch: {num_training_seq_per_batch}, "
              f"iter_samples_per_batchID: {len(self.labelled_flag)}, "
              f"total_iter_samples(include unlabelled seq): {len(indices)}")
        return indices, random_seeds

    def _re_indices_test_fixLen(self):
        assert self.test_seq_len is not None
        self.training_seq_len = self.test_seq_len
        print("test mode, repeat to fix seq length. start re-indices!")

        # split original seq to training_seq
        training_seq_list = []
        for seq_name, idxs in self.seq_dict.items():
            labelled_samples = np.where(idxs[:, 2] == 1)[0]
            training_seq_end_idxs = labelled_samples
            training_seq_se_idxs = [(max(-1, end - self.test_seq_len), end) for end in training_seq_end_idxs]
            training_seq_data_idxs = [list(idxs[range(start + 1, end + 1), 1]) for start, end in training_seq_se_idxs]
            for i in range(len(training_seq_data_idxs)):
                if len(training_seq_data_idxs[i]) < self.test_seq_len:
                    training_seq_data_idxs[i] = [idxs[0, 1]] * (self.test_seq_len - len(training_seq_data_idxs[i])) + \
                                                training_seq_data_idxs[i]
            assert np.all(np.array([len(data_idx) == self.test_seq_len for data_idx in training_seq_data_idxs]))
            training_seq_list.extend(training_seq_data_idxs)

        if self.shuffle:
            np.random.shuffle(training_seq_list)

        # use training_seq_list to assemble batch!
        ## check the last seq in a batch
        base_i = 0
        indices = []
        for bs in range(self.batch_size):
            indices_per_batch = []
            for i in range(self.samples_per_batchID):
                new_i = base_i + i
                if new_i >= len(training_seq_list):
                    new_i = new_i - len(training_seq_list)
                indices_per_batch.extend(training_seq_list[new_i])
            base_i += self.samples_per_batchID
            indices.append(indices_per_batch)

        labelled_template = [0] * self.test_seq_len
        labelled_template[-1] = 1
        self.labelled_flag = labelled_template * self.samples_per_batchID
        ## check! the labelled samples in resembled seqs should be equal to samples_per_batchID
        assert sum(self.labelled_flag) == self.samples_per_batchID

        indices = np.array(indices).transpose().reshape(-1)

        indices = indices.astype(np.int64).tolist()
        self.total_iter_samples = len(indices)

        print(f"ep: {self.epoch}, test_seq_len: {self.test_seq_len}, "
              f"labelled_samples_per_batchID: {self.samples_per_batchID}, "
              f"total_labelled_samples: {self.num_samples}, "
              f"num_training_seq_per_batch: {self.samples_per_batchID}, "
              f"iter_samples_per_batchID: {len(self.labelled_flag)}, "
              f"total_iter_samples(include unlabelled seq): {len(indices)}")
        return indices

    def _re_indices_test(self):
        assert self.test_seq_len is not None
        print("test mode, no repeat. start re-indices!")

        self.fixed_training_seq_len = self.test_seq_len
        indices, _ = self._re_indices(test=True)
        return indices

    def __iter__(self):
        if not self.test_mode:
            indices, random_seeds = self._re_indices()
            out_path = 'random_seeds.pkl'
            with open(out_path, 'wb') as f:
                pickle.dump(random_seeds, f)
        else:
            if self.repeat_test:
                indices = self._re_indices_test_fixLen()  # repeated to ensure each labelled frame has test_len seq
            else:
                indices = self._re_indices_test()
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def total_iter_samples(self):
        return self.total_iter_samples


class DistributedSeqSampler(SeqSampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, max_training_seq=100, samples_per_gpu=1, max_epoches=20, shuffle=False,
                 num_replicas=None, rank=None, test_seq_len=None, training_seq_len=None, repeat_test=True):

        super().__init__(dataset, max_training_seq=max_training_seq, samples_per_gpu=samples_per_gpu,
                         max_epoches=max_epoches, shuffle=shuffle, test_seq_len=test_seq_len,
                         training_seq_len=training_seq_len, repeat_test=repeat_test)

        _rank, _num_replicas = get_dist_info()
        print('sampler: rank:{}, world_size:{}'.format(_rank, _num_replicas))
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank

        self.batch_size = self.samples_per_gpu * self.num_replicas

        self.samples_per_batchID = int(
            np.ceil(self.labeled_samples / self.batch_size))  # total samples for one batch_id
        self.num_samples_per_replica = self.samples_per_batchID * self.samples_per_gpu
        self.num_samples = self.samples_per_batchID * self.batch_size

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        if not self.test_mode:
            indices, random_seeds = self._re_indices()
        else:
            if self.repeat_test:
                indices = self._re_indices_test_fixLen()  # repeated to ensure each labelled frame has test_len seq
            else:
                indices = self._re_indices_test()

        # subsample
        offset = self.rank
        indices = indices[offset:len(indices):self.num_replicas]
        if not self.test_mode:
            out_path = f'random_seeds_{self.rank}.pkl'
            new_random_seeds = {}
            for idx in indices:
                new_random_seeds[idx] = random_seeds[idx]
            with open(out_path, 'wb') as f:
                pickle.dump(new_random_seeds, f)

        return iter(indices)

    def __len__(self):
        return self.num_samples_per_replica

    def set_epoch(self, epoch):
        self.epoch = epoch
