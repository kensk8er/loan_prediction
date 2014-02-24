#!/usr/local/bin/python

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import scipy.stats as stats
import sklearn.linear_model as lm


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


def create_sub(classifier, train_data, labels, test_data):
    sub = 1

    labels = np.asarray(map(int, labels))

    niter = 10
    auc_list = []
    mean_auc = 0.0
    itr = 0

    if sub == 1:
        x_train = train_data  # [train]
        x_test = test_data  # [test]

        y_train = labels  # [train]

        predicts_orig = np.asarray([0] * test_data.shape[0])  # np.copy(y_test)

        predicts_p = pd.read_csv("data/test_pred_bin_hansong.csv")

        none_zero_train = np.where(y_train > 0)[0]
        #none_zero_test = np.where(predicts_p > 0)[0]
        none_zero_test = np.where(predicts_p['loss'] == True)[0]

        zero_train = np.where(y_train == 0)[0]
        #zero_test = np.where(predicts_p == 0)[0]
        zero_test = np.where(predicts_p['loss'] == False)[0]

        # use only train data which are classified as non zero loss
        x_train_p = x_train[none_zero_train]
        x_test_p = x_test[none_zero_test]

        # y_train is loss values (not binary)
        y_train_0 = y_train[zero_train]
        y_train_1 = y_train[none_zero_train]

        # grouping loss values
        count = [0 for i in range(12)]
        for i in xrange(len(y_train_1)):
            if y_train_1[i] <= 10:
                count[y_train_1[i]-1] += 1
                y_train_1[i] = y_train_1[i] * 0.5 * 10
            elif y_train_1[i] < 100:
                count[10] += 1
                y_train_1[i] = 11 * 0.5 * 10
            else:
                count[11] += 1
                y_train_1[i] = 10 * 10


        print count
        print 'fitting loss values using logistic regression...'  # this process takes the longest time
        classifier.fit(x_train_p, y_train_1)
        print 'accuracy rate:'
        print classifier.score(x_train_p, y_train_1)

        print 'predicting loss for test data...'
        predicts = classifier.predict(x_test_p)

        predicts_orig[none_zero_test] = predicts
        predicts_orig[zero_test] = 0

        report = []
        for i in xrange(len(predicts_orig)):
            report.append(float(predicts_orig[i]) / 10.)

        print 'writing result...'
        np.savetxt('result/predictions.csv', report, delimiter=',', fmt='%s')


if __name__ == '__main__':
    print 'loading test data...'
    X_test = test_data('data/test_best_30.csv')

    print 'loading train data...'
    X, labels = train_data('data/train_best_30.csv')

    print 'define logistic regression...'
    classifier = lm.LogisticRegression(penalty='l2', dual=True, tol=0.00001,
                                       C=0.8, fit_intercept=True, intercept_scaling=1.0,
                                       random_state=None, class_weight={5: 0.012, 10: 0.035, 15: 0.038, 20: 0.051, 25: 0.064, 30: 0.077, 35: 0.09, 40: 0.1, 45: 0.11, 50: 0.13, 55: 0.14, 100: 0.15})
    #classifier = lm.LinearRegression(fit_intercept=True, normalize=False)

    print 'pre-processing train data...'
    scalar = preprocessing.StandardScaler().fit(X)
    X = scalar.transform(X)
    #X = preprocessing.scale(X)

    print 'pre-processing test data...'
    #X_test = preprocessing.scale(X_test)
    X_test = scalar.transform(X_test)

    create_sub(classifier, X, labels, X_test)
