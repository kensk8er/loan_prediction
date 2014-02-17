__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
# add a path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../util')

import time

import numpy

import theano
import theano.tensor as T

from unpickle import unpickle
import csv

from logistic_sgd import LogisticRegression, load_data, save_parameters


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #		activation function used (among other things).
        #		For example, results presented in [Xavier10] suggest that you
        #		should use 4 times larger initial weights for sigmoid
        #		compared to tanh
        #		We have no info for other function, so we use the same as
        #		tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                  + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                      + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100000,
             dataset='cifar-10-batches-py', batch_size=20, test_batch_size=32, n_hidden=700, mode='train',
             amount='full', valid_num=10000):  #batch_size: 32

    datasets = load_data(dataset, mode, amount, valid_num)

    if mode == 'train':
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
    else:
        test_set_x, test_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    if mode == 'train':
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    else:
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / test_batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=769,
                     n_hidden=n_hidden, n_out=101)

    ## load the saved parameters
    if mode == 'test':
        learned_params = unpickle('params/mlp.pkl')


    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
           + L1_reg * classifier.L1 \
           + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    if mode == 'test':
        test_model = theano.function(inputs=[index],
                                     outputs=classifier.errors(y),
                                     givens={
                                     x: test_set_x[index * test_batch_size: (index + 1) * test_batch_size],
                                     y: test_set_y[index * test_batch_size: (index + 1) * test_batch_size]})
    else:
        validate_model = theano.function(inputs=[index],
                                         outputs=classifier.errors(y),
                                         givens={
                                         x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                         y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

        train_error_model = theano.function(inputs=[index],
                                            outputs=classifier.errors(y),
                                            givens={
                                            x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                            y: train_set_y[index * batch_size:(index + 1) * batch_size]})

        get_train_labels = theano.function([index], classifier.logRegressionLayer.ex_y,
                                           givens={
                                           x: train_set_x[index * batch_size: (index + 1) * batch_size]})

    if mode == 'test':
        get_test_labels = theano.function([index], classifier.logRegressionLayer.ex_y,
                                          givens={
                                          x: test_set_x[index * test_batch_size: (index + 1) * test_batch_size],
                                          classifier.hiddenLayer.W: learned_params[0],
                                          classifier.hiddenLayer.b: learned_params[1],
                                          classifier.logRegressionLayer.W: learned_params[2],
                                          classifier.logRegressionLayer.b: learned_params[3]})


    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    if mode == 'train':
        gparams = []
        for param in classifier.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        updates = []
        # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #	C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        for param, gparam in zip(classifier.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(inputs=[index], outputs=cost,
                                      updates=updates,
                                      givens={
                                      x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    #init_bias = [-1. for i in xrange(101)]
    ##init_bias = numpy.asarray(init_bias, dtype=numpy.float64)
    #init_bias[0] = 100.
    #initialize_bias = theano.function(inputs=[], outputs=classifier.logRegressionLayer.b,
    #		updates={classifier.logRegressionLayer.b: init_bias},
    #		givens={classifier.logRegressionLayer.b: init_bias})

    #bias = initialize_bias()
    #print bias


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.999  # a relative improvement of this much is
    # considered significant
    if mode == 'train':
        validation_frequency = min(n_train_batches, patience / 2)
        # go through this many
        # minibatche before checking the network
        # on the validation set; in this case we
        # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    if mode == 'train':
        done_looping = False
    else:
        done_looping = True

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                train_losses = [train_error_model(i)
                                for i in xrange(n_train_batches)]
                this_train_loss = numpy.mean(train_losses)

                try:
                    pred_labels = variable
                except NameError:
                    pred_labels = [[0 for j in xrange(batch_size)] for i in xrange(n_train_batches)]

                #params = get_params()
                #print 'W[0:10]:', params[0][0:10], 'b[0:10]:', params[1][0:10]

                if mode == 'train':
                    for i in xrange(n_train_batches):
                        pred_labels[i] = get_train_labels(i)

                    #print 'max predicted labels:',
                    #for i in xrange(len(pred_labels)):
                    #	print max(pred_labels[i]),
                    #print

                print('epoch %i, minibatch %i/%i, validation error (MAE) %f' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss))

                print('epoch %i, minibatch %i/%i, training error (MAE) %f' % \
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_train_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    ## save the parameters
                    get_params = theano.function(inputs=[], outputs=[classifier.hiddenLayer.W, classifier.hiddenLayer.b,
                                                                     classifier.logRegressionLayer.W,
                                                                     classifier.logRegressionLayer.b])
                    save_parameters(get_params(), 'mlp')

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break

    if mode == 'train':
        for i in xrange(n_train_batches):
            pred_labels[i] = get_train_labels(i)

        print 'max predicted labels:',
        for i in xrange(len(pred_labels)):
            print max(pred_labels[i]),
        print

    if mode == 'test':
        print 'predicting the labels...'
        pred_labels = [[0 for j in xrange(batch_size)] for i in xrange(n_test_batches)]
        for i in xrange(n_test_batches):
            print str(i + 1), '/', str(n_test_batches)
            pred_labels[i] = get_test_labels(i)

        writer = csv.writer(file('result/mlp.csv', 'w'))
        writer.writerow(['id', 'loss'])
        row = 105472  # first ID of test data

        print 'output test labels...'
        for i in xrange(len(pred_labels)):
            print str(i + 1), '/', str(len(pred_labels))
            for j in xrange(len(pred_labels[i])):
                writer.writerow([row, pred_labels[i][j]])
                row += 1

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f '
           'obtained at iteration %i') %
          (best_validation_loss, best_iter + 1))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        test_mlp()
    else:
        if args[1] == 'test':
            mode_ = 'test'
        else:
            mode_ = 'train'
        if len(args) > 2:
            if args[2] == 'min':
                amount_ = 'min'
            else:
                amount_ = 'full'
            test_mlp(mode=mode_, amount=amount_)
        else:
            test_mlp(mode=mode_)

