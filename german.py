import numpy as np
import os
from urllib import parse, request
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


class GermanPrepare:
    def __init__(self, target_dir, download_dir, n_splits=5, val_split=0.3):
        """
        target_dir: where we want to save pre-processed data into pkl
        download_dir: where we want to download original data
        """
        self.source_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/'
        self.file_names = ['german.' + name for name in ['data']]
        self.target_dir = target_dir
        self.download_dir = download_dir
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)

        self.n_splits = n_splits
        self.val_split = val_split
        self.col_names = ['status', 'duration', 'credit-history', 'purpose', 'credit-amount', 'savings', 'employment',
                          'installment-rate', 'sex', 'debtors', 'residence', 'property', 'age', 'installment-plan',
                          'housing', 'exist-credit', 'job', 'liable-people', 'telephone', 'foreign', 'label']

    def download(self):
        for filename in self.file_names:
            src_url = parse.urljoin(self.source_url, filename)
            download_uri = os.path.join(self.download_dir, filename)
            print(f'downloading to {download_uri}...')
            request.urlretrieve(src_url, download_uri)

    def prepare(self):
        file = os.path.join(self.download_dir, 'german.data')

        x = pd.read_csv(file, sep=' ', names=self.col_names)
        self._clean(x)

        y = x.pop('label')
        y[y == 1] = 0
        y[y == 2] = 1

        tele = x.pop('telephone')
        tele[tele == 'A191'] = 0
        tele[tele == 'A192'] = 1

        foreign = x.pop('foreign')
        foreign[foreign == 'A201'] = 0
        foreign[foreign == 'A202'] = 1

        categorical_cols = ['status', 'credit-history', 'purpose',
                            'savings', 'employment', 'sex', 'debtors',
                            'property', 'installment-plan', 'housing', 'job']
        categorical = x[categorical_cols]

        sex = categorical.pop('sex').apply(lambda text: text.strip())
        sex[sex == 'A92'] = 0
        sex[sex == 'A93'] = 1
        sex[sex == 'A91'] = 1
        sex[sex == 'A94'] = 1
        numeric = x.drop(columns=categorical_cols)

        one_hot_categorical = pd.get_dummies(categorical)
        purpose_a47 = pd.DataFrame(0, index=np.arange(1000), columns=['purpose_A47'])

        one_hot_numeric = pd.concat([pd.cut(numeric[name], 5) for name in numeric], axis=1)
        one_hot_numeric.columns = numeric.columns
        one_hot_numeric = pd.concat([pd.get_dummies(one_hot_numeric[name], prefix=one_hot_numeric[name].name)
                                     for name in one_hot_numeric], axis=1)

        one_hot = pd.concat([one_hot_categorical.iloc[:, :11], one_hot_categorical.iloc[:, 12:17],
                             purpose_a47, one_hot_categorical.iloc[:, 17:19],
                             one_hot_categorical.iloc[:, 11], one_hot_categorical.iloc[:, 19:],
                             tele, foreign, one_hot_numeric, sex, y], axis=1)
        train, test = train_test_split(one_hot, test_size=300)
        train.index = range(700)
        test.index = range(300)
        train.to_pickle(os.path.join(self.target_dir, 'german_train.pkl'))
        test.to_pickle(os.path.join(self.target_dir, 'german_test.pkl'))

    @staticmethod
    def _clean(dataset):
        dataset.replace(' ?', np.nan, inplace=True)
        dataset.dropna(inplace=True)
        dataset.index = range(len(dataset))