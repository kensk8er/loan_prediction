#!/usr/local/bin/python

import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
print 'iris.data.shape', iris.data.shape
print 'iris.target.shape', iris.target.shape

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)

print 'X_train:', X_train.shape
print 'y_train: ', y_train.shape
print 'X_test: ', X_test.shape
print 'y_test: ', y_test.shape

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#print 'score:', clf.score(X_test, y_test)

scores = cross_validation.cross_val_score(
    clf, iris.data, iris.target, cv=5, scoring='accuracy')

print scores

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

n_samples = iris.data.shape[0]
cv = cross_validation.ShuffleSplit(n_samples, n_iter=3,
                                   test_size=0.3, random_state=0)

scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

