import numpy as np
import os
from urllib import parse, request
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


class AdultPrepare:
    def __init__(self, target_dir, download_dir, n_splits=5, val_split=0.3):
        """
        target_dir: where we want to save pre-processed data into pkl
        download_dir: where we want to download original data
        """
        self.source_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
        self.file_names = ['adult.' + name for name in ['data', 'test']]
        self.target_dir = target_dir
        self.download_dir = download_dir
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)

        self.n_splits = n_splits
        self.val_split = val_split
        self.col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                          'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                          'native-country',
                          'label']

    def download(self):
        for filename in self.file_names:
            src_url = parse.urljoin(self.source_url, filename)
            download_uri = os.path.join(self.download_dir, filename)
            print(f'downloading to {download_uri}...')
            request.urlretrieve(src_url, download_uri)

    def prepare(self):
        train_file = os.path.join(self.download_dir, 'adult.data')
        test_file = os.path.join(self.download_dir, 'adult.test')

        train = pd.read_csv(train_file, sep=',', names=self.col_names)
        test = pd.read_csv(test_file, sep=',', names=self.col_names, skiprows=[0])
        self._clean(train)
        self._clean(test)
        x = pd.concat([train, test])

        y = x.pop('label').apply(lambda text: text.strip().strip('.'))
        y[y == '<=50K'] = 0
        y[y == '>50K'] = 1

        sex = x.pop('sex')
        sex[sex == 'Female'] = 0
        sex[sex == 'Male'] = 1

        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                            'native-country']
        categorical = x[categorical_cols]
        numeric = x.drop(columns=categorical_cols)

        age = numeric.pop('age')
        age[age < 65] = 0
        age[age >= 65] = 1

        one_hot_categorical = pd.get_dummies(categorical)
        one_hot_num = pd.concat([pd.qcut(numeric[name], 2, duplicates='drop').astype('category').cat.codes for name in numeric], axis=1)
        one_hot_num.columns = numeric.columns
        one_hot = pd.concat([one_hot_categorical, sex, one_hot_num, age, y], axis=1)
        one_hot[0:len(train)].to_pickle(os.path.join(self.target_dir, 'adult_train.pkl'))
        one_hot[len(train):].to_pickle(os.path.join(self.target_dir, 'adult_test.pkl'))

    @staticmethod
    def _clean(dataset):
        dataset.replace(' ?', np.nan, inplace=True)
        dataset.dropna(inplace=True)
        dataset.index = range(len(dataset))