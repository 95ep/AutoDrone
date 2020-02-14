import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils

"""
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
"""
import random
import itertools

class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = data
        #self.observations = data['obs']
        #self.actions = data['act']
        #self.rewards = data['rew']

        #self.length = self.actions.shape[0]
        self.length = self.data['act'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #return self.observations[idx], self.actions[idx], self.rewards[idx]
        return {k: v[idx] for k, v in self.data.items()}


class CustomSampler(Sampler):

    def __init__(self, data_source, minibatch_size, shuffle=True, drop_last=True):
        self.data_source = data_source
        self.minibatch_size = minibatch_size
        self.drop_last = drop_last
        self.length = len(data_source)
        self.shuffle = shuffle

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



if __name__ == '__main__':

    obs = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13]).view(7,2)
    act = torch.tensor([0,1,2,3,4,5,6]).view(7,1)
    rew = torch.tensor([-6,-5,-4,-3,-2,-1,0]).view(7,1)
    data = {'obs': obs, 'act': act, 'rew': rew}

    dataset = CustomDataset(data)
    sampler = CustomSampler(dataset, 3, drop_last=False)
    data_loader = DataLoader(dataset, batch_size=3, sampler=sampler)

    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch, sample_batched)
