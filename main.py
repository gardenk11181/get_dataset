import os
import pickle
from german import GermanPrepare
from adult import AdultPrepare
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataset import Adult, German
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import warnings


def experiment(type):
    """


    :param type: 0 -> german, 1 -> adult, 2 -> health
    :return: scores of logistic, random forest, random choice & accuracy on y
    """
    if type == 0:
        sensitive = 'sex'
        target_dir = './german/'
        with open(os.path.join(target_dir, 'german_train.pkl'), 'rb') as f:
            train = pickle.load(f)
        with open(os.path.join(target_dir, 'german_test.pkl'), 'rb') as f:
            test = pickle.load(f)
    elif type == 1:
        sensitive = 'age'
        target_dir = './adult/'
        with open(os.path.join(target_dir, 'adult_train.pkl'), 'rb') as f:
            train = pickle.load(f)
        with open(os.path.join(target_dir, 'adult_test.pkl'), 'rb') as f:
            test = pickle.load(f)
    else:
        sensitive = 'age'

    train_s = train.pop(sensitive).astype('category')
    train_y = train.pop('label').astype('category')
    train_x = train.astype('category')

    test_s = test.pop(sensitive).astype('category')
    test_y = test.pop('label').astype('category')
    test_x = test.astype('category')

    # logistic regression
    log = LogisticRegression(random_state=1).fit(train_x, train_s)
    log_score = log.score(test_x, test_s)

    # random forest
    rf = RandomForestClassifier(random_state=1).fit(train_x, train_s)
    rf_score = rf.score(test_x, test_s)

    # random choice
    random = DummyClassifier(random_state=1, strategy='uniform').fit(train_x, train_s)
    random_score = random.score(test_x, test_s)

    # accuracy on y
    log_y = LogisticRegression(random_state=1).fit(train_x, train_y)
    log_y_score = log_y.score(test_x, test_y)

    # discrimination on s
    pred = pd.Series(log_y.predict(test_x), dtype='category')
    s_zero = test_s == 0
    s_one = test_s == 1
    disc = np.abs(((pred[s_zero] == test_y[s_zero]).sum()) / s_zero.sum() -
                  ((pred[s_one] == test_y[s_one]).sum()) / s_one.sum())

    # discrimination prob on s
    pred_probs = log_y.predict_proba(test_x)
    pred_prob = pd.Series(np.apply_along_axis(lambda x: x[0] if x[0] > x[1] else x[1], 1, pred_probs), dtype=float)
    disc_prob = np.abs((pred_prob[s_zero].sum()) / s_zero.sum() -
                       (pred_prob[s_one].sum()) / s_one.sum())

    return log_score, rf_score, random_score, disc, disc_prob, log_y_score


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    dict = {0: 'german', 1: 'adult'}
    for i in range(2):
        print('---------------------------------------------------------------------------')
        print(dict[i])
        print('---------------------------------------------------------------------------')
        log_score, rf_score, random_score, disc, disc_prob, log_y_score = experiment(i)
        print('logistic: %f, random forest: %f, random choice: %f' %
              (log_score, rf_score, random_score))
        print('discrimination on s: %f, dsicrimination prob on s: %f' % (disc, disc_prob))
        print('accuracy on y: %f' % log_y_score)
