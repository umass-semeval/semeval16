import numpy as np
import theano
import theano.tensor as T
from LogisticRegression import *

"""
copied from http://deeplearning.net/tutorial/code/mlp.py
"""


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None else activation(lin_output)
        )
        self.params = [self.W, self.b]


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hidden_layer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)
        self.log_regression_layer = LogisticRegression(input=self.hidden_layer.output, n_in=n_hidden, n_out=n_out)
        self.L1 = (abs(self.hidden_layer.W).sum() + abs(self.log_regression_layer.W).sum())
        self.L2_sqr = ((self.hidden_layer.W ** 2).sum() + (self.log_regression_layer.W ** 2).sum())
        self.nll = self.log_regression_layer.nll
        self.errors = self.log_regression_layer.errors
        self.make_shared_dataset = self.log_regression_layer.make_shared_dataset
        self.params = self.hidden_layer.params + self.log_regression_layer.params
        self.input = input
        self.ypred = self.log_regression_layer.ypred


def train_mlp(train_x, train_y, dev_x, dev_y, nclasses,
              learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001,
              nepochs=100, batch_size=10, n_hidden=100):

    ndims = train_x.shape[1]
    print("ndims: %d, nclasses: %d" % (ndims, nclasses))

    n_train_batches = train_x.shape[0] / batch_size
    print("n train batches: %d" % n_train_batches)
    n_dev_batches = dev_x.shape[0] / batch_size

    print('building model')
    index = T.lscalar()  # index to a minibatch
    x = T.matrix('x')
    y = T.ivector('y')
    rng = np.random.RandomState(1234)

    classifier = MLP(rng=rng, input=x, n_in=ndims, n_hidden=n_hidden, n_out=nclasses)

    train_x, train_y = classifier.make_shared_dataset(train_x, train_y)

    dev_x_orig = dev_x[:]
    dev_y_orig = dev_y[:]

    dev_x, dev_y = classifier.make_shared_dataset(dev_x, dev_y)

    cost = classifier.nll(y) + L1_reg*classifier.L1 + L2_reg*classifier.L2_sqr
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: dev_x[index * batch_size: (index+1) * batch_size],
            y: dev_y[index * batch_size: (index+1) * batch_size]
        })

    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size],
        })

    print("training...")
    patience = 5000
    patience_incr = 2
    improve_threshold = 0.995
    valid_freq = min(n_train_batches, patience / 2)
    best_valid_loss = np.inf
    valid_score = -1.0
    start_time = timeit.default_timer()

    done = False
    epoch = 0
    while epoch < nepochs and not done:
        epoch = epoch + 1
        for mbatch_idx in xrange(n_train_batches):
            mbatch_avg_cost = train_model(mbatch_idx)
            iter = (epoch - 1) * n_train_batches + mbatch_idx
            if (iter + 1) % valid_freq == 0:
                valid_losses = [test_model(i) for i in xrange(n_dev_batches)]
                this_vloss = np.mean(valid_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, mbatch_idx + 1, n_train_batches, this_vloss * 100.))
                if this_vloss < best_valid_loss:
                    if this_vloss < best_valid_loss * improve_threshold:
                        patience = max(patience, iter * patience_incr)
                    best_valid_loss = this_vloss
                    valid_score = this_vloss
                # TODO the following doesnt work for some reason:
                # with open('best_model.pkl', 'w') as f:
                #     cPickle.dump(classifier, f)
                if patience <= iter:
                    done = True
                    break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_valid_loss * 100., valid_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    return predict2(classifier, dev_x_orig, dev_y_orig)


def predict2(classifier, test_x, test_y):
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.ypred)
    # test_x, test_y = classifier.make_shared_dataset(test_x, test_y)
    pred_values = predict_model(test_x)
    n = test_x.shape[0]
    ncorrect = 0.0
    total = 0.0
    for i in range(n):
        pred = pred_values[i]
        truth = test_y[i]
        if pred == truth:
            ncorrect += 1.0
        total += 1.0
    return ncorrect, total
