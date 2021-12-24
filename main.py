import os
import pickle
from adult import AdultPrepare

if __name__ == '__main__':
    target_dir = './adult/'
    source_dir = './adult_source/'
    adult = AdultPrepare(target_dir, source_dir)
    # adult.download()
    adult.prepare()
    train_path = os.path.join(target_dir, 'adult_test.pkl')
    with open(train_path, 'rb') as f:
        df = pickle.load(f)
    print(df.columns)
