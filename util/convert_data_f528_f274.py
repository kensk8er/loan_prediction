#!/usr/local/bin/python

import sys
import os
# add a path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../util')

import csv
from pickle import pickle
import numpy
from var_dump import var_dump


def normalize(file_names):
    ''' normalize each value between 0 and 1 '''

    row_lens = {}

    for file_name in file_names:
        print 'normalizing the data... (%s)' % file_name
        csvfile = open('data/' + file_name + '.csv')
        row_num = 0

        for row in csv.reader(csvfile):
            row_num += 1
            print '\r', row_num,

            # obtaining normalizing factors only from training data
            if file_name == file_names[0]:
                #print row[522], row[272]
                if row_num >= 2:  # first row is the name of columns
                    try:
                        if isinstance(min_val, float) and isinstance(max_val, float):
                            # compare values if already defined
                            # for column in xrange(len(row) - 1):
                            #     if min_val[column] > float(row[column + 1]):
                            #         min_val[column] = float(row[column + 1])
                            #
                            #     if max_val[column] < float(row[column + 1]):
                            #         max_val[column] = float(row[column + 1])
                            if row[522] == 'NA' or row[272] == 'NA':
                                row[522] = 0
                                row[272] = 0

                            if min_val > float(row[522]) - float(row[272]):
                                min_val = float(row[522]) - float(row[272])

                            if max_val < float(row[522]) - float(row[272]):
                                max_val = float(row[522]) - float(row[272])

                    except NameError:
                        # initialize min_val and max_val if not defined
                        #min_val = [0 for i in xrange(len(row) - 1)]  # 1-st column is an ID of data (irrelevant)
                        #max_val = [0 for i in xrange(len(row) - 1)]  # 1-st column is an ID of data (irrelevant)

                        # first assignment
                        #for column in xrange(len(row) - 1):
                        min_val = float(row[522]) - float(row[272])
                        max_val = float(row[522]) - float(row[272])

        csvfile.close()
        print
        row_lens[file_name] = row_num

    return max_val, min_val, row_lens


def train(file_name, row_len):

    train_set_x = []
    train_set_y = []

    print 'converting the data... (%s)' % file_name
    csvfile = open('data/' + file_name + '.csv')
    row_num = 0

    for row in csv.reader(csvfile):
        row_num += 1

        print '\r', row_num, '/', row_len,
        #features = row[1: len(row) - 1]  # 1-st column is an ID of data (irrelevant)
        #print features[521], features[271]

        if row_num != 1:
            features = row[1: len(row) - 1]  # 1-st column is an ID of data (irrelevant)

            # for column in xrange(len(features)):
            #     if max_val[column] != min_val[column]:
            #         features[column] = (float(features[column]) - min_val[column]) / (max_val[column] - min_val[column])
            #     else:
            #         # TBF: if max_val == min_val, that feature is no use for prediction, thus better removing the column
            #         features[column] = 0.
            feature = []
            if features[521] == 'NA' or features[271] == 'NA':
                features[521] = 0
                features[271] = 0

            value = float(features[521]) - float(features[271])
            feature.append(value)
            #feature.append((value - min_val) / (max_val - min_val))

            train_set_x.append(feature)

            loss = row[-1]
            # if int(loss) > 0:
            #     loss = 1
            # else:
            #     loss = 0

            train_set_y.append(int(loss))

    csvfile.close()
    train_set_x = numpy.asarray(train_set_x, dtype=numpy.float64)
    train_set = (train_set_x, train_set_y)

    print
    print 'saving the data...'
    pickle(train_set, 'data/' + file_name + '_f528_f274.pkl')
    print 'done!'


def test(file_name, row_len):

    print 'converting the data... (%s)' % file_name
    csvfile = open('data/' + file_name + '.csv')

    ## convert test data
    test_set = []
    row_num = 0

    for row in csv.reader(csvfile):
        row_num += 1

        print '\r', row_num, '/', row_len,

        if row_num != 1:
            features = row[1: len(row)]  # 1-st column is an id of data (irrelevant)

            # for column in xrange(len(features)):
            #     if max_val[column] != min_val[column]:
            #         features[column] = (float(features[column]) - min_val[column]) / (max_val[column] - min_val[column])
            #     else:
            #         # TBF: if max_val == min_val, that feature is no use for prediction, thus better removing the column
            #         features[column] = 0.

            feature = []

            if features[521] == 'NA' or features[271] == 'NA':
                features[521] = 0
                features[271] = 0

            value = float(features[521]) - float(features[271])
            feature.append(value)
            #feature.append((value - min_val) / (max_val - min_val))

            test_set.append(feature)


    csvfile.close()
    test_set = numpy.asarray(test_set, dtype=numpy.float64)

    print
    print 'saving the data...'
    pickle(test_set, 'data/' + file_name + '_f528_f274.pkl')
    print 'done!'


if __name__ == '__main__':
    args = sys.argv

    if len(args) > 2:
        file_names = (args[1], args[2])

        # calculate normalize parameters
        max_val, min_val, row_lens = normalize(file_names)
        #max_val = 0
        #min_val = 0
        #row_lens = {}
        #row_lens[file_names[0]] = 30

        # generate train and valid set
        train(file_names[0], row_lens[file_names[0]])
        # generate test set
        test(file_names[1], row_lens[file_names[1]])

    else:
        print 'input train, valid, and test file names as arguments!'
