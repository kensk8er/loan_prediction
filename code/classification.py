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

        print 'prepare all zero labels...'
        predicts_orig = np.asarray([0] * test_data.shape[0])  # np.copy(y_test)

        # perhaps you can delete these lines
        labels_p = []

        for i in xrange(len(labels)):
            if labels[i] > 0:
                labels_p.append(1)
            else:
                labels_p.append(0)

        labels_p = np.asarray(labels_p)
        y_train_p = labels_p

        # use already classified entries
        #print 'select features using Linear SVC...'
        #linear_svc = LinearSVC(C=0.01, penalty="l1", dual=False, verbose=2)  # C: Penalty parameter C of the error term
        #linear_svc.fit(x_train, y_train_p)
        #x_train_p = linear_svc.transform(x_train)  # transform(X[, threshold])	Reduce X to its most important features.
        #x_test_p = linear_svc.transform(x_test)

        #print 'classify default entries using logistic regression...'
        #classifier.fit(x_train_p, y_train_p)
        #predicts_p = classifier.predict(x_test_p)
        predicts_p = pd.read_csv("data/test_pred_bin.csv")

        none_zero_train = np.where(y_train_p > 0)[0]
        #none_zero_test = np.where(predicts_p > 0)[0]
        none_zero_test = np.where(predicts_p['loss'] == True)[0]

        zero_train = np.where(y_train_p == 0)[0]
        #zero_test = np.where(predicts_p == 0)[0]
        zero_test = np.where(predicts_p['loss'] == False)[0]

        # use only train data which are classified as non zero loss
        x_train_p = x_train[none_zero_train]
        x_test_p = x_test[none_zero_test]

        # y_train is loss values (not binary)
        y_train_0 = y_train[zero_train]
        y_train_1 = y_train[none_zero_train]

        print 'fitting loss values using logistic regression...'  # this process takes the longest time
        classifier.fit(x_train_p, y_train_1)
        predicts = classifier.predict(x_test_p)

        predicts_orig[none_zero_test] = predicts
        predicts_orig[zero_test] = 0

        print 'writing result...'
        np.savetxt('result/predictions.csv', predicts_orig, delimiter=',', fmt='%d')


if __name__ == '__main__':
    print 'loading test data...'
    test_x= test_data('data/test_20_best_classify.csv')

    print 'loading train data...'
    train_x, train_y = train_data('data/train_20_best_classify.csv')

    print 'define logistic regression...'
    classifier = lm.LogisticRegression(penalty='l2', dual=False, tol=0.00001,
                                       C=10000, fit_intercept=True, intercept_scaling=1.0, # C=10000
                                       random_state=None, class_weight={0: 0.135, 1: 0.865})
    #classifier = lm.LinearRegression(fit_intercept=True, normalize=False)

    print 'pre-processing train data...'
    scalar = preprocessing.StandardScaler().fit(train_x)
    train_x = scalar.transform(train_x)
    #X = preprocessing.scale(X)

    print 'pre-processing test data...'
    test_x = scalar.transform(test_x)
    #X_test = preprocessing.scale(X_test)
    #X_test = scalar.transform(X_test)

    #create_sub(classifier, X, labels, X_test)

    # perhaps you can delete these lines
    labels_p = []

    for i in xrange(len(train_y)):
        if train_y[i] > 0:
            labels_p.append(1)
        else:
            labels_p.append(0)

    labels_p = np.asarray(labels_p, dtype=int)
    train_y_p = labels_p

    print 'doing logistic regression...'
    classifier.fit(train_x, train_y_p)
    print classifier.score(train_x, train_y_p)

    predicts = classifier.predict(test_x)
    np.savetxt('result/classification.csv', predicts, delimiter=',', fmt='%s')