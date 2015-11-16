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


def sgd_optimization(train_x, train_y, dev_x, dev_y, nclasses, learning_rate=0.01, nepochs=100, batch_size=10):
    ndims = train_x.shape[1]
    print("ndims: %d, nclasses: %d" % (ndims, nclasses))

    n_train_batches = train_x.shape[0] / batch_size
    print("n train batches: %d" % n_train_batches)
    n_dev_batches = dev_x.shape[0] / batch_size

    print('building model')
    index = T.lscalar()  # index to a minibatch
    x = T.matrix('x')
    y = T.ivector('y')
    classifier = LogisticRegression(input=x, n_in=ndims, n_out=nclasses)

    train_x, train_y = classifier.make_shared_dataset(train_x, train_y)
    dev_x, dev_y = classifier.make_shared_dataset(dev_x, dev_y)

    cost = classifier.nll(y)
    test_model = theano.function(
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
                with open('best_model.pkl', 'w') as f:
                    cPickle.dump(classifier, f)
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


def predict(test_x, test_y):
    classifier = cPickle.load(open('best_model.pkl'))
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.ypred)
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
