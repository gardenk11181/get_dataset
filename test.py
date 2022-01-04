from experiment import preprocess, calculate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
from adult import AdultPrepare
from german import GermanPrepare
import pandas as pd
import numpy as np
import pickle

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')

    german = GermanPrepare('german', 'german_source')
    german.prepare()

    dict = {0: 'german', 1: 'adult', 2: 'health'}
    dict_model = {0: 'x', 1: 'vae', 2: 'vfae'}
    for i in [0]:
        z, s, y = preprocess(i)
        [log_scores, rf_scores, log_y_scores, disc_scores, disc_prob_scores] = calculate(z, s, y)
        print('---------------------------------------------------------------------------')
        print(dict[i])
        print('---------------------------------------------------------------------------')
        for j in range(1):
            print('     %s' % dict_model[j])
            print('     logistic: %f, random forest: %f' % (log_scores[j], rf_scores[j]))
            print('     discrimination on s: %f, discrimination prob on s: %f' % (disc_scores[j], disc_prob_scores[j]))
            print('     accuracy on y: %f' % log_y_scores[j])
            print('---------------------------------------------------------------------------')
