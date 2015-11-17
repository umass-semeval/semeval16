import numpy as np
import theano
import theano.tensor as T
import timeit
import cPickle

"""
copied from http://deeplearning.net/tutorial/code/logistic_sgd.py
"""


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='W',
            borrow=True)
        self.b = theano.shared(
            value=np.zeros((n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True)
        self.prob_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.ypred = T.argmax(self.prob_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

        self.num_in = n_in
        self.num_out = n_out

    def nll(self, y):
        """
        mean negative log likelihood of prediction
        """
        return -T.mean(T.log(self.prob_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        assert y.ndim == self.ypred.ndim, "y shape should match ypred shape"
        assert y.dtype.startswith('int'), "not implemented?"
        return T.mean(T.neq(self.ypred, y))

    def make_shared_dataset(self, data_x, data_y, borrow=True):
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')


def logistic_regression_optimization_sgd(train, dev, test, nclasses, learning_rate=0.01, nepochs=100, batch_size=10):
    train_x, train_y = train
    dev_x, dev_y = dev
    test_x, test_y = test

    ndims = train_x.shape[1]
    print("ndims: %d, nclasses: %d" % (ndims, nclasses))

    print('building model')
    index = T.lscalar()  # index to a minibatch
    x = T.matrix('x')
    y = T.ivector('y')
    classifier = LogisticRegression(input=x, n_in=ndims, n_out=nclasses)

    train_x, train_y = classifier.make_shared_dataset(train_x, train_y)
    dev_x, dev_y = classifier.make_shared_dataset(dev_x, dev_y)
    test_x, test_y = classifier.make_shared_dataset(test_x, test_y)

    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
    print("n train batches: %d" % n_train_batches)
    n_dev_batches = dev_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_x.get_value(borrow=True).shape[0] / batch_size

    cost = classifier.nll(y)

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

    # compute gradients, define update
    gW = T.grad(cost=cost, wrt=classifier.W)
    gb = T.grad(cost=cost, wrt=classifier.b)
    updates = [(classifier.W, classifier.W - learning_rate * gW),
               (classifier.b, classifier.b - learning_rate * gb)]

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
    patience = 5000
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

                    with open('best_model.pkl', 'w') as f:
                        cPickle.dump(classifier, f)

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


def predict(test_x, test_y):
    classifier = cPickle.load(open('best_model.pkl'))
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.ypred)
    pred_values = predict_model(test_x)
    return pred_values
