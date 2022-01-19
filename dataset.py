import torch
import os
import pickle5 as pickle
import numpy as np
import pandas as pd

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
            tmp = pickle.load(r)[self.tp]
            if self.tp != 1:
                self.x = pd.DataFrame(tmp[0].todense())
                self.y = tmp[1]
            else:
                self.x = pd.DataFrame(tmp.todense())
            if self.tp == 0:
                self.s = torch.as_tensor(0, dtype=torch.float32).reshape(-1)
            else:
                self.s = torch.as_tensor(1, dtype=torch.float32).reshape(-1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x.iloc[index]
        if self.tp == 1:
            x = torch.as_tensor(x, dtype=torch.float32)
            # x: [batch_size, 5000], s: [batch_size, 1]
            return [x, self.s]

        else:
            y = torch.as_tensor(self.y.iloc[index], dtype=torch.float32)
            x = torch.as_tensor(x, dtype=torch.float32)
            # x: [batch_size, 5000], s: [batch_size, 1], y: [batch_size, 2]
            return [x, self.s, y]


class mYaleB(Dataset):
    def __init__(self, dir_path, train=True, label=True):
        if train:
            self.path = os.path.join(dir_path, 'eyaleb_train.pkl')
        else:
            self.path = os.path.join(dir_path, 'eyaleb_test.pkl')
        with open(self.path, 'rb') as f:
            self.data = pickle.load(f)

        self.code = torch.from_numpy(np.eye(38)).type(torch.float32)
        self.code2 = torch.from_numpy(np.eye(5)).type(torch.float32)

    def __len__(self):
        return (self.data['data'].shape)[0]

    def __getitem__(self, idx):
        y = self.code[self.data['label'][idx].astype(np.int)]
        s = self.code2[self.data['light'][idx].astype(np.int)]
        x = torch.from_numpy((2. * (self.data['data'][idx] / 255) - 1.).reshape((1, 168, 192))).type(torch.float32)
        target = torch.from_numpy(self.data['label'][idx].reshape(-1)).type(torch.float32)

        # x: [batch_size, 168*192], s: [batch_size, 5], y: [batch_size, 38], target: [batch_size, 1]
        return [x, s, y, target]
