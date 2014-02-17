#!/usr/local/bin/python

import sys
import os
# add a path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../util')

import csv
#from unpickle import unpickle
from pickle import pickle
import numpy


def main(filename, mode):

    # normalize each value between 0 and 1
    print 'normalizing the data...'
    csvfile = open('data/' + filename + '.csv')
    row_num = 0

    for row in csv.reader(csvfile):
        row_num += 1
        print "\r", row_num,

        if row_num == 1:  # initialize
            min_val = [0 for i in xrange(len(row) - 1)]  # 1-st column is an ID of data (irrelevant)
            max_val = [0 for i in xrange(len(row) - 1)]  # 1-st column is an ID of data (irrelevant)
        elif row_num == 2:  # first assignment
            for column in xrange(len(row) - 1):
                min_val[column] = float(row[column + 1])
                max_val[column] = float(row[column + 1])
        else:
            for column in xrange(len(row) - 1):
                if min_val[column] > float(row[column + 1]):
                    min_val[column] = float(row[column + 1])

                if max_val[column] < float(row[column + 1]):
                    max_val[column] = float(row[column + 1])

    row_len = row_num

    if mode != 'test':
        train_set_x = []
        train_set_y = []

        print 'converting the data...'
        csvfile = open('data/' + filename + '.csv')
        row_num = 0

        for row in csv.reader(csvfile):
            row_num += 1

            print row_num, '/', row_len

            if row_num != 1:
                features = row[1: len(row) - 1]  # 1-st column is an ID of data (irrelevant)

                for i in xrange(len(features)):
                    if max_val[i] != min_val[i]:
                        features[i] = (float(features[i]) - min_val[i]) / (max_val[i] - min_val[i])
                    else:
                        features[i] = 0.

                train_set_x.append(features)

                loss = row[-1]
                train_set_y.append(int(loss))

        train_set_x = numpy.asarray(train_set_x, dtype=numpy.float64)
        train_set = (train_set_x, train_set_y)

        print 'saving the data...'
        pickle(train_set, 'data/' + filename + '.pkl')
    else:
        print 'converting the data...'
        csvfile = open('data/' + filename + '.csv')
        row_num = 0

        ## convert test data
        test_set = []

        row_num = 0

        for row in csv.reader(csvfile):
            row_num += 1

            print row_num, '/', row_len

            if row_num != 1:
                features = row[1: len(row)]  # 1-st column is an ID of data (irrelevant)

                for i in xrange(len(features)):
                    if max_val[i] != min_val[i]:
                        features[i] = (float(features[i]) - min_val[i]) / (max_val[i] - min_val[i])
                    else:
                        features[i] = 0.

                test_set.append(features)

        test_set = numpy.asarray(test_set, dtype=numpy.float64)

        print 'saving the data...'
        pickle(test_set, 'data/' + filename + '.pkl')

    csvfile.close()


if __name__ == '__main__':
    argvs = sys.argv
    if len(argvs) > 2:
        filename = argvs[1]
        mode = argvs[2]
        main(filename, mode)
    else:
        print 'input file name and mode as an argument!'
