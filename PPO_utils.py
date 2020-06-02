import itertools
import numpy as np
import random
import scipy.signal
import torch
from torch.utils.data import Dataset, Sampler


def discount_cumsum(x, discount):
    """
    Calculate a cummulative sum
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer:
    """
    Stores collected trajectories of experience.
    """
    def __init__(self, steps, vector_shape, visual_shape, action_dim, gamma, lam):
        # Constructor - set xxx_shape to None if not used
        if vector_shape is not None:
            self.obs_vector = np.zeros((steps, *vector_shape), dtype=np.float32)
        if visual_shape is not None:
            self.obs_visual = np.zeros((steps, *visual_shape), dtype=np.float32)
        if action_dim > 1:
            self.act_buf = np.zeros((steps, action_dim), dtype=np.float32)
        else:
            self.act_buf = np.zeros((steps, action_dim), dtype=np.int)
        self.rew_buf = np.zeros(steps, dtype=np.float32)
        self.adv_buf = np.zeros((steps, 1), dtype=np.float32)
        self.ret_buf = np.zeros((steps, 1), dtype=np.float32)
        self.val_buf = np.zeros(steps, dtype=np.float32)
        self.logp_buf = np.zeros((steps, 1), dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0  # Keeps track of number of interactions in buffer
        self.path_start_idx = 0
        self.max_size = steps

    def store(self, obs_vector, obs_visual, act, rew, val, logp):
        """
        Add one step of interactions to the buffer, set obs_xx to None if not used
        """
        assert self.ptr < self.max_size
        if hasattr(self, 'obs_vector'):
            self.obs_vector[self.ptr] = obs_vector
        if hasattr(self, 'obs_visual'):
            self.obs_visual[self.ptr] = obs_visual
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call at the end of a trajectory to calculate the correct advantages and returns.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = np.expand_dims(discount_cumsum(deltas, self.gamma * self.lam), axis=1)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = np.expand_dims(discount_cumsum(rews, self.gamma)[:-1], axis=1)

        self.path_start_idx = self.ptr

    def get(self):
        """
        Return all data from the buffer in the form of torch tensors
        """
        assert self.ptr == self.max_size  # buffer must be full
        self.ptr, self.path_start_idx = 0, 0

        # If you wish to use advantage normalization, uncomment following 3 lines.
        # adv_mean = np.mean(self.adv_buf)
        # adv_std = np.std(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf)
        if hasattr(self, 'obs_vector'):
            data['obs_vector'] = self.obs_vector
        if hasattr(self, 'obs_visual'):
            data['obs_visual'] = self.obs_visual

        # Convert into torch tensors
        data = {k: torch.as_tensor(v) for k, v in data.items()}

        return data


class ExperienceDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.length = self.data['act'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        experience_dict = {k: v[idx] for k, v in self.data.items()}
        return experience_dict


class ExperienceSampler(Sampler):

    def __init__(self, data_source, minibatch_size, shuffle=True, drop_last=True):
        self.data_source = data_source
        self.minibatch_size = minibatch_size
        self.drop_last = drop_last
        self.length = len(data_source)
        self.shuffle = shuffle
        super().__init__(data_source)

    def __len__(self):
        return self.length

    def __iter__(self):
        num_batches = self.length // self.minibatch_size
        order = [[*range(i, i+self.minibatch_size)]
                 for i in range(0, self.minibatch_size * num_batches, self.minibatch_size)]

        if self.shuffle:
            random.shuffle(order)

        if not self.drop_last and self.minibatch_size * num_batches < self.length:
            order.append([*range(self.minibatch_size * num_batches, self.length)])

        return itertools.chain.from_iterable(order)
