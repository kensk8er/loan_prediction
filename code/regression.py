#!/usr/local/bin/python

import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.svm import LinearSVC
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


def do_regression(classifier, data, mode):

    if mode == 'test':
        x_train = data[0]
        y_train = np.asarray(map(int, data[1]))
        x_test = data[1]
    else:
        x_train = data[0]
        y_train = np.asarray(map(int, data[1]))

    if mode == 'test':
        predicts_orig = np.asarray([0] * x_test.shape[0])
        predicted_bin = pd.read_csv("data/test_pred_bin_hansong.csv")

    # possibly you can try using train data which are classified as default by your classification model??
    none_zero_train = np.where(y_train > 0)[0]

    if mode == 'test':
        none_zero_test = np.where(predicted_bin['loss'] == True)[0]

    zero_train = np.where(y_train == 0)[0]

    if mode == 'test':
        zero_test = np.where(predicted_bin['loss'] == False)[0]

    # use only train data which are classified as non zero loss
    x_train_default = x_train[none_zero_train]

    if mode == 'test':
        x_test_default = x_test[none_zero_test]

    # y_train is loss values (not binary)
    y_train_non_default = y_train[zero_train]
    y_train_default = y_train[none_zero_train]

    # grouping loss values (black magic lol)
    count = [0 for i in range(12)]
    for i in xrange(len(y_train_default)):
        if y_train_default[i] <= 10:
            count[y_train_default[i]-1] += 1
            y_train_default[i] = y_train_default[i] * 0.5 * 10
        elif y_train_default[i] < 100:
            count[10] += 1
            y_train_default[i] = 11 * 0.5 * 10
        else:
            count[11] += 1
            y_train_default[i] = 10 * 10

    #print count

    if mode == 'test':
        print 'fitting loss values using logistic regression...'

        classifier.fit(x_train_default, y_train_default)
        accuracy = classifier.score(x_train_default, y_train_default)
        print 'accuracy on the training data', accuracy

        print 'predicting loss for test data...'
        predicts = classifier.predict(x_test_default)

        predicts_orig[none_zero_test] = predicts
        predicts_orig[zero_test] = 0

        report = []
        for i in xrange(len(predicts_orig)):
            report.append(float(predicts_orig[i]) / 10.)

        print 'writing result...'
        np.savetxt('result/predictions.csv', report, delimiter=',', fmt='%s')
    else:
        cv = 5
        print cv, 'fold cross validation...'

        scores = cross_validation.cross_val_score(
            classifier, x_train_default, y_train_default, cv=cv, scoring='accuracy')

        print("Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
        # my best: Accuracy: 0.27609578 (+/- 0.00854811) (5 fold cross validation)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        args = sys.argv
        mode = args[1]
    else:
        mode = 'validate'

    print 'mode:', mode

    if mode == 'test':
        print 'loading test data...'
        X_test = test_data('data/test_best_30.csv')

    print 'loading train data...'
    X, labels = train_data('data/train_best_30.csv')

    print 'define logistic regression...'
    classifier = lm.LogisticRegression(penalty='l2', dual=True, tol=0.00001,
                                       C=0.8, fit_intercept=True, intercept_scaling=1.0,
                                       random_state=None, class_weight={5: 0.012, 10: 0.035, 15: 0.038, 20: 0.051, 25: 0.064, 30: 0.077, 35: 0.09, 40: 0.1, 45: 0.11, 50: 0.13, 55: 0.14, 100: 0.15})

    print 'pre-processing train data...'
    scalar = preprocessing.StandardScaler().fit(X)
    X = scalar.transform(X)

    if mode == 'test':
        print 'pre-processing test data...'
        X_test = scalar.transform(X_test)

        do_regression(classifier, (X, labels, X_test), mode)
    else:
        do_regression(classifier, (X, labels), mode)
