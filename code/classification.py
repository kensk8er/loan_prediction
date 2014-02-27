#!/usr/local/bin/python
from numpy.ma import mean

import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import scipy.stats as stats
import sklearn.linear_model as lm
import sys


def test_data(filename):
    X = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)
    X = np.asarray(X.values, dtype=float)

    col_mean = stats.nanmean(X, axis=0)
    ids = np.where(np.isnan(X))
    X[ids] = np.take(col_mean, ids[1])
    data = np.asarray(X[:, 1:], dtype=float)  # the first column is ID.

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
        test_x = test_data('data/test_classify.csv')

    print 'loading train data...'
    train_x, train_y = train_data('data/train_classify.csv')

    print 'defining logistic regression...'
    #classifier = lm.LogisticRegression(penalty='l2', dual=False, tol=0.00001,
    #                                   C=10000, fit_intercept=True, intercept_scaling=1.0, # C=10000
    #                                   random_state=None, class_weight={0: 0.135, 1: 0.865})
    classifier = SVC(tol=0.01, C=10, #gamma=0.48, # default: 0.25
                     random_state=None, class_weight={0: 0.135, 1: 0.865})

    print 'pre-processing train data...'
    scalar = preprocessing.StandardScaler().fit(train_x)
    train_x = scalar.transform(train_x)

    if mode == 'test':
        print 'pre-processing test data...'
        test_x = scalar.transform(test_x)


    train_y_bin = []

    for i in xrange(len(train_y)):
        if train_y[i] > 0:
            train_y_bin.append(1)
        else:
            train_y_bin.append(0)

    train_y_bin = np.asarray(train_y_bin, dtype=int)

    print 'doing logistic regression...'

    if mode == 'test':
        classifier.fit(train_x, train_y_bin)
        accuracy = classifier.score(train_x, train_y_bin)
        print 'accuracy on the training data', accuracy

        predicts = classifier.predict(test_x)

        print 'writing result...'
        np.savetxt('result/classification.csv', predicts, delimiter=',', fmt='%s')
    else:
        cv = 5
        print cv, 'fold cross validation...'

        scores = cross_validation.cross_val_score(
            classifier, train_x, train_y_bin, cv=cv, scoring='accuracy')

        print("Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2))
        # my best: Accuracy: 0.98128395 (5-fold cross validation)
        # SVC: 0.98099002

        classifier.fit(train_x, train_y_bin)
        predicts = classifier.predict(train_x)
        predicts = np.asarray(predicts, dtype=int)
        none_zero = np.where(predicts == 1)[0]

        none_zero_y = train_y[none_zero]
        average = mean(none_zero_y)
        print 'average loss of predicted records:', average
