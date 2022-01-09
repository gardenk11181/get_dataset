import torch
import os
import pickle5 as pickle

from torch.utils.data import Dataset


class Adult(Dataset):
    def __init__(self, dir_path, train=True):
        if train:
            self.path = os.path.join(dir_path, 'adult_train.pkl')
        else:
            self.path = os.path.join(dir_path, 'adult_test.pkl')
        with open(self.path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        y = torch.as_tensor(data.pop('label'), dtype=torch.float32).reshape(-1)
        s = torch.as_tensor(data.pop('sex'), dtype=torch.float32).reshape(-1)
        x = torch.as_tensor(data, dtype=torch.float32)

        # x: [batch_size, 121], s&y: [batch_size, 1]
        return [x, s, y]


class German(Dataset):
    def __init__(self, dir_path, train=True):
        if train:
            self.path = os.path.join(dir_path, 'german_train.pkl')
        else:
            self.path = os.path.join(dir_path, 'german_test.pkl')
        with open(self.path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        y = torch.as_tensor(data.pop('label'), dtype=torch.float32).reshape(-1)
        s = torch.as_tensor(data.pop('sex'), dtype=torch.float32).reshape(-1)
        x = torch.as_tensor(data, dtype=torch.float32)

        # x: [batch_size, 84], s&y: [batch_size, 1]
        return [x, s, y]


class Health(Dataset):
    def __init__(self, dir_path, train=True):
        if train:
            self.path = os.path.join(dir_path, 'health_train.pkl')
        else:
            self.path = os.path.join(dir_path, 'health_test.pkl')
        with open(self.path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        y = torch.as_tensor(data.pop('label'), dtype=torch.float32).reshape(-1)
        s = torch.as_tensor(data.pop('age'), dtype=torch.float32).reshape(-1)
        x = torch.as_tensor(data, dtype=torch.float32)

        # x: [batch_size, 621], s&y: [batch_size, 1]
        return [x, s, y]


class Amazon(Dataset):
    def __init__(self, dir_path, source, target, tp=0):
        """
        :param dir_path: ~/amazon
        :param source: source domain e.g. books, dvd
        :param target: target domain e.g. electronics, kitchen
        :param tp: 0 -> source train, 1 -> target train, 2-> target test
        """
        self.path = os.path.join(dir_path, '%s_to_%s.pkl' % (source, target))
        self.tp = tp
        with open(self.path, 'rb') as r:
            self.data = pickle.load(r)[self.tp]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        if self.tp == 1:
            x = torch.as_tensor(data, dtype=torch.float32)
            # x: [batch_size, 5000]
            return [x]

        else:
            y = torch.as_tensor(data.pop('y'), dtype=torch.float32).reshape(-1)
            x = torch.as_tensor(data, dtype=torch.float32)
            # x: [batch_size, 5000], y: [batch_size, 1]
            return [x, y]
