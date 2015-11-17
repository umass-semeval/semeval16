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

        self.num_in = n_in
        self.num_out = n_out


class MLP(object):
    def __init__(self, rng, input, n_in, n_out, n_layers, units):
        #  build and connect each hidden layer
        assert n_layers == len(units)
        self.hidden_layers = []
        self.hidden_layers.append(HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=units[0], activation=T.tanh))
        i = 1
        while i < n_layers:
            prev_layer = self.hidden_layers[-1]
            layer = HiddenLayer(rng=rng, input=prev_layer.output, n_in=units[i-1], n_out=units[i], activation=T.tanh)
            self.hidden_layers.append(layer)
            i += 1

        # logistic regression layer for classification
        self.softmax_layer = LogisticRegression(input=self.hidden_layers[-1].output, n_in=units[-1], n_out=n_out)

        self.L1 = sum([abs(self.hidden_layers[i].W).sum() for i in xrange(n_layers)]) + abs(self.softmax_layer.W).sum()
        self.L2_sqr = sum([(self.hidden_layers[i].W ** 2).sum() for i in xrange(n_layers)]) + \
                      (self.softmax_layer.W ** 2).sum()
        self.nll = self.softmax_layer.nll
        self.errors = self.softmax_layer.errors
        self.make_shared_dataset = self.softmax_layer.make_shared_dataset

        self.params = self.hidden_layers[0].params
        for i in xrange(1, n_layers):
            self.params += self.hidden_layers[i].params
        self.params += self.softmax_layer.params

        self.input = input
        self.ypred = self.softmax_layer.ypred


def build_mlp(ndims, nclasses, nlayers, L1_reg, L2_reg, units):
    print('building model')
    index = T.lscalar()  # index to a minibatch
    x = T.matrix('x')
    y = T.ivector('y')
    rng = np.random.RandomState(1234)
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=ndims,
        n_out=nclasses,
        n_layers=nlayers,
        units=units
    )
    cost = (
        classifier.nll(y) + L1_reg*classifier.L1 + L2_reg*classifier.L2_sqr
    )
    print("classifier with %d hidden layers" % len(classifier.hidden_layers))
    return classifier, cost, index, x, y


def train_mlp(train, dev, test, nclasses,
              learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001,
              nlayers=2, units=[800, 400],
              nepochs=100, batch_size=10):

    train_x, train_y = train
    dev_x, dev_y = dev
    test_x, test_y = test

    ndims = train_x.shape[1]
    print("ndims: %d, nclasses: %d" % (ndims, nclasses))

    classifier, cost, index, x, y = build_mlp(ndims, nclasses, nlayers, L1_reg, L2_reg, units)

    train_x, train_y = classifier.make_shared_dataset(train_x, train_y)
    dev_x, dev_y = classifier.make_shared_dataset(dev_x, dev_y)
    test_x, test_y = classifier.make_shared_dataset(test_x, test_y)

    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
    print("n train batches: %d" % n_train_batches)
    n_dev_batches = dev_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_x.get_value(borrow=True).shape[0] / batch_size

    # cost = (
    #     classifier.nll(y) + L1_reg*classifier.L1 + L2_reg*classifier.L2_sqr
    # )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_x[index * batch_size: (index + 1) * batch_size],
            y: test_y[index * batch_size: (index + 1) * batch_size]
        })

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: dev_x[index * batch_size: (index+1) * batch_size],
            y: dev_y[index * batch_size: (index+1) * batch_size]
        })

    # # compute gradients, define update
    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)
    ]

    # define training procedure
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size],
        })

    """
    MODEL TRAINING
    """
    print("training...")
    # early stopping params
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < nepochs) and (not done_looping):
        epoch += 1
        for batch_idx in xrange(n_train_batches):
            batch_avg_cost = train_model(batch_idx)
            iter = (epoch - 1) * n_train_batches + batch_idx

            if (iter + 1) % validation_frequency == 0:
                dev_losses = [validate_model(i) for i in xrange(n_dev_batches)]
                this_dev_loss = np.mean(dev_losses)
                print('epoch %i, batch %i/%i, validation error %f %%' % (epoch, batch_idx + 1, n_train_batches, this_dev_loss*100.))
                if this_dev_loss < best_validation_loss:
                    if this_dev_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_dev_loss

                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print('epoch %i, batch %i/%i, test error of best model %f %%' % (epoch, batch_idx+1, n_train_batches, test_score*100.))

                    # with open('best_model.pkl', 'w') as f:
                    #     cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        ('optimization complete with best validation score of %f %%,'
         'with test performance %f %%'
         ) % (best_validation_loss*100., test_score*100.)
    )
    print('ran for %d epochs, with %f epochs/sec' % (epoch, 1.*epoch/(end_time - start_time)))
    return classifier


def predict2(classifier, test_x, test_y):
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.ypred)
    # test_x, test_y = classifier.make_shared_dataset(test_x, test_y)
    pred_values = predict_model(test_x)
    return pred_values
