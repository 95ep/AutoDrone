import random
import itertools
from torch.utils.data import Dataset, Sampler


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
    