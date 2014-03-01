#!/usr/local/bin/python
from numpy.ma import mean

import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC
import scipy.stats as stats
import sklearn.linear_model as lm
import sys


def test_data(filename):
    X = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)
    X = np.asarray(X.values, dtype=float)

    col_mean = stats.nanmean(X, axis=0)
    ids = np.where(np.isnan(X))
    X[ids] = np.take(col_mean, ids[1])
    data = np.asarray(X[:, 1:], dtype=float)  # why 1:-3?? -> last two features are use less. the first column is ID.

    return data


def train_data(filename):
    X = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)
    X = np.asarray(X.values, dtype=float)

    col_mean = stats.nanmean(X, axis=0)
    ids = np.where(np.isnan(X))
    X[ids] = np.take(col_mean, ids[1])  # ids[0]: row index, ids[1]: column index

    labels = np.asarray(X[:, -1], dtype=float)
    data = np.asarray(X[:, 1:-1], dtype=float)
    return data, labels

if __name__ == '__main__':

    if len(sys.argv) > 1:
        args = sys.argv
        mode = args[1]
    else:
        mode = 'validate'

    print 'mode:', mode

    if mode == 'test':
        print 'loading test data...'
        x_test = test_data('data/test_default.csv')

    print 'loading train data...'
    x_train, y_train = train_data('data/train_default.csv')

    # possibly you can try using train data which are classified as default by your classification model??
    none_zero_train = np.where(y_train > 0)[0]

    # if mode == 'test':
    #     none_zero_test = np.where(predicted_bin['loss'] == True)[0]

    zero_train = np.where(y_train == 0)[0]

    # if mode == 'test':
    #     zero_test = np.where(predicted_bin['loss'] == False)[0]

    # use only train data which are classified as non zero loss
    x_train_default = x_train[none_zero_train]

    # if mode == 'test':
    #     x_test_default = x_test[none_zero_test]

    # y_train is loss values (not binary)
    y_train_non_default = y_train[zero_train]
    y_train_default = y_train[none_zero_train]

    print 'pre-processing train data...'
    scalar = preprocessing.StandardScaler().fit(x_train_default)
    x_train = scalar.transform(x_train_default)

    # if mode == 'test':
    #     print 'pre-processing test data...'
    #     x_test = scalar.transform(x_test)

    for k in range(1,50):
        classifier = KMeans(n_clusters=k, n_init=10, max_iter=300, tol=0.0001, precompute_distances=True)

        classifier.fit(x_train_default)
        score = classifier.score(x_train_default)
        print k, 'cluster:', score

        predicts = classifier.predict(x_train_default)
        count = {}
        loss_value = [[] for i in range(k)]

        for i in xrange(len(predicts)):
            count[predicts[i]] = count.get(predicts[i], 0) + 1
            loss_value[predicts[i]].append(y_train_default[i])  # = loss_value.get(predicts[i], 0) + y_train_default[i]

        for i in range(len(loss_value)):
            mean = np.mean(loss_value[i])
            std = np.std(loss_value[i])
            print 'cluster', i, ': count =', count[i], ', mean =', mean, ', std =', std
