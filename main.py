import os
import pickle
from german import GermanPrepare
from adult import AdultPrepare
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == '__main__':
    target_dir = './german/'
    source_dir = './german_source/'
    # adult = GermanPrepare(target_dir, source_dir)
    # adult.download()
    # adult.prepare()
    with open(os.path.join(target_dir, 'german_test.pkl'), 'rb') as f:
        df = pickle.load(f)
    pd.set_option('max_columns', None)
    df[['duration', 'credit-amount', 'installment-rate', 'residence', 'age', 'exist-credit', 'liable-people']].hist(figsize=(20,20))
    plt.show()

# if __name__ == '__main__':
#     target_dir = './adult/'
#     source_dir = './adult_source/'
#     adult = AdultPrepare(target_dir, source_dir)
#     # adult.download()
#     adult.prepare()
#     with open(os.path.join(target_dir, 'adult_train.pkl'), 'rb') as f:
#         df = pickle.load(f)
#     plt.show()


