import os
import pickle5 as pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def preprocess(tp, dir_path):
    """
    :param dir_path: ~/VFAE
    :param tp: 0 -> german, 1 -> adult, 2 -> health
    :return: scores of logistic, random forest, random choice & accuracy on y
    """
    if tp == 0:
        data = 'german'
        sensitive = 'sex'
        target_dir = dir_path + '/dataset/german/'
        with open(os.path.join(target_dir, 'german_train.pkl'), 'rb') as f:
            train = pickle.load(f)
        with open(os.path.join(target_dir, 'german_test.pkl'), 'rb') as f:
            test = pickle.load(f)
    elif tp == 1:
        data = 'adult'
        sensitive = 'sex'
        target_dir = dir_path + '/dataset/adult/'
        with open(os.path.join(target_dir, 'adult_train.pkl'), 'rb') as f:
            train = pickle.load(f)
        with open(os.path.join(target_dir, 'adult_test.pkl'), 'rb') as f:
            test = pickle.load(f)
    else:
        data = 'health'
        target_dir = dir_path + '/dataset/health/'
        sensitive = 'age'
        with open(os.path.join(target_dir, 'health_train.pkl'), 'rb') as f:
            train = pickle.load(f)
        with open(os.path.join(target_dir, 'health_test.pkl'), 'rb') as f:
            test = pickle.load(f)

    train_z1_vae = np.load(os.path.join(dir_path + '/exp_%s/' % data, "%s_train_z1_vae.npy" % data))
    test_z1_vae = np.load(os.path.join(dir_path + '/exp_%s/' % data, "%s_test_z1_vae.npy" % data))
    train_z1_vfae = np.load(os.path.join(dir_path + '/exp_%s/' % data, "%s_train_z1_vfae.npy" % data))
    test_z1_vfae = np.load(os.path.join(dir_path + '/exp_%s/' % data, "%s_test_z1_vfae.npy" % data))

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

    log_scores = []
    rf_scores = []
    log_y_scores = []
    disc_scores = []
    disc_prob_scores = []
    for i in range(3):
        # logistic regression & random forest
        log = LogisticRegression(random_state=1).fit(train_z1[i], train_s)
        log_scores.append(log.score(test_z1[i], test_s))

        rf = RandomForestClassifier(random_state=1).fit(train_z1[i], train_s)
        rf_scores.append(rf.score(test_z1[i], test_s))

        log_y = LogisticRegression(random_state=1).fit(train_z1[i], train_y)
        log_y_scores.append(log_y.score(test_z1[i], test_y))

        # discrimination on s
        pred = log_y.predict(test_z1[i])
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
