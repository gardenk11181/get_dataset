import os
import pickle5 as pickle
from german import GermanPrepare
from adult import AdultPrepare
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
        data = 'german'
        sensitive = 'sex'
        target_dir = './german/'
        with open(os.path.join(target_dir, 'german_train.pkl'), 'rb') as f:
            train = pickle.load(f)
        with open(os.path.join(target_dir, 'german_test.pkl'), 'rb') as f:
            test = pickle.load(f)
    elif type == 1:
        data = 'adult'
        sensitive = 'age'
        target_dir = './adult/'
        with open(os.path.join(target_dir, 'adult_train.pkl'), 'rb') as f:
            train = pickle.load(f)
        with open(os.path.join(target_dir, 'adult_test.pkl'), 'rb') as f:
            test = pickle.load(f)
    else:
        data = 'health'
        target_dir = './health/'
        sensitive = 'age'
        with open(os.path.join(target_dir, 'health_train.pkl'), 'rb') as f:
            train = pickle.load(f)
        with open(os.path.join(target_dir, 'health_test.pkl'), 'rb') as f:
            test = pickle.load(f)

    train_z1_vae = np.load(os.path.join('./DM2020', "%s_train_z1_vae.npy" % data))
    test_z1_vae = np.load(os.path.join('./DM2020', "%s_test_z1_vae.npy" % data))
    train_z1_vfae = np.load(os.path.join('./DM2020', "%s_train_z1_vfae.npy" % data))
    test_z1_vfae = np.load(os.path.join('./DM2020', "%s_test_z1_vfae.npy" % data))

    train_s = train.pop(sensitive).astype('category')
    train_y = train.pop('label').astype('category')
    train_x = train.astype('category')

    test_s = test.pop(sensitive).astype('category')
    test_y = test.pop('label').astype('category')
    test_x = test.astype('category')

    train_z1 = [train_x, train_z1_vae, train_z1_vfae]
    test_z1 = [test_x, test_z1_vae, test_z1_vfae]

    return [train_z1, test_z1], [train_s, test_s], [train_y, test_y]

def calculate(z, s, y):
    [train_z1, test_z1] = z
    [train_s, test_s] = s
    [train_y, test_y] = y
    # logistic regression
    log_scores = []
    rf_scores = []
    log_y_scores = []
    disc_scores = []
    disc_prob_scores = []
    for i in range(3):
        log = LogisticRegression(random_state=1).fit(train_z1[i], train_s)
        log_scores.append(log.score(test_z1[i], test_s))

        rf = RandomForestClassifier(random_state=1, bootstrap=False).fit(train_z1[i], train_s)
        rf_scores.append(rf.score(test_z1[i], test_s))

        log_y = LogisticRegression(random_state=1).fit(train_z1[i], train_y)
        log_y_scores.append(log_y.score(test_z1[i], test_y))

        # discrimination on s
        pred = pd.Series(log_y.predict(test_z1[i]), dtype='category')
        s_zero = test_s == 0
        s_one = test_s == 1
        disc_scores.append(np.abs(((pred[s_zero] == test_y[s_zero]).sum()) / s_zero.sum() -
                                  ((pred[s_one] == test_y[s_one]).sum()) / s_one.sum()))

        # discrimination prob on s
        pred_probs = log_y.predict_proba(test_z1[i])
        pred_prob = pd.Series(np.apply_along_axis(lambda x: x[0] if x[0] > x[1] else x[1], 1, pred_probs), dtype=float)
        disc_prob_scores.append(np.abs((pred_prob[s_zero].sum()) / s_zero.sum() -
                                       (pred_prob[s_one].sum()) / s_one.sum()))

    return [log_scores, rf_scores, log_y_scores, disc_scores, disc_prob_scores]


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    dict = {0: 'german', 1: 'adult', 2: 'health'}
    dict_model = {0: 'x', 1: 'vae', 2: 'vfae'}
    for i in [1]:
        z, s, y = experiment(i)
        [log_scores, rf_scores, log_y_scores, disc_scores, disc_prob_scores] = calculate(z, s, y)
        print('---------------------------------------------------------------------------')
        print(dict[i])
        print('---------------------------------------------------------------------------')
        for j in range(3):
            print('     %s' % dict_model[j])
            print('     logistic: %f, random forest: %f' % (log_scores[j], rf_scores[j]))
            print('     discrimination on s: %f, discrimination prob on s: %f' % (disc_scores[j], disc_prob_scores[j]))
            print('     accuracy on y: %f' % log_y_scores[j])
            print('---------------------------------------------------------------------------')
