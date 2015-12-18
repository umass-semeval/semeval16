# -*- coding: utf-8 -*-
import cPickle
import os
import time
import argparse
from collections import OrderedDict
import numpy as np

import theano
import theano.tensor as T
import lasagne
import lasagne.layers as layer
from lasagne.layers import get_output_shape

from evaluate import ConfusionMatrix
from hparams import HParams

MAXLEN = 140
SEED = 1234


def build_model(hyparams,
                vmap,
                log,
                nclasses=2,
                batchsize=None,
                invar=None,
                maskvar=None,
                maxlen=MAXLEN):

    embedding_dim = hyparams.embedding_dim
    nhidden = hyparams.nhidden
    bidirectional = hyparams.bidirectional
    pool = hyparams.pool
    grad_clip = hyparams.grad_clip

    net = OrderedDict()

    V = len(vmap)
    W = lasagne.init.Normal()

    gate_params = layer.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(),
        W_hid=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.)
    )

    cell_params = layer.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(),
        W_hid=lasagne.init.Orthogonal(),
        W_cell=None,
        b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.tanh
    )

    net['input'] = layer.InputLayer((batchsize, maxlen), input_var=invar)
    net['mask'] = layer.InputLayer((batchsize, maxlen), input_var=maskvar)
    ASSUME = {net['input']: (200, 140), net['mask']: (200, 140)}
    net['emb'] = layer.EmbeddingLayer(net['input'], input_size=V, output_size=embedding_dim, W=W)
    net['fwd1'] = layer.LSTMLayer(
        net['emb'],
        num_units=nhidden,
        grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh,
        mask_input=net['mask'],
        ingate=gate_params,
        forgetgate=gate_params,
        cell=cell_params,
        outgate=gate_params,
        learn_init=True
    )
    if bidirectional:
        net['bwd1'] = layer.LSTMLayer(
            net['emb'],
            num_units=nhidden,
            grad_clipping=grad_clip,
            nonlinearity=lasagne.nonlinearities.tanh,
            mask_input=net['mask'],
            ingate=gate_params,
            forgetgate=gate_params,
            cell=cell_params,
            outgate=gate_params,
            learn_init=True,
            backwards=True
        )
        if pool == 'mean':
            def tmean(a, b):
                agg = theano.tensor.add(a, b)
                agg /= 2.
                return agg
            net['pool'] = layer.ElemwiseMergeLayer([net['fwd1'], net['bwd1']], tmean)
        elif pool == 'sum':
            net['pool'] = layer.ElemwiseSumLayer([net['fwd1'], net['bwd1']])
        else:
            net['pool'] = layer.ConcatLayer([net['fwd1'], net['bwd1']])
    else:
        net['pool'] = layer.ConcatLayer([net['fwd1']])
    net['dropout1'] = layer.DropoutLayer(net['pool'], p=0.5)
    net['fwd2'] = layer.LSTMLayer(
        net['dropout1'],
        num_units=nhidden,
        grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh,
        mask_input=net['mask'],
        ingate=gate_params,
        forgetgate=gate_params,
        cell=cell_params,
        outgate=gate_params,
        learn_init=True,
        only_return_final=True
    )
    net['dropout2'] = layer.DropoutLayer(net['fwd2'], p=0.6)
    net['softmax'] = layer.DenseLayer(
        net['dropout2'],
        num_units=nclasses,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    logstr = '========== MODEL ========== \n'
    logstr += 'vocab size: %d\n' % V
    logstr += 'embedding dim: %d\n' % embedding_dim
    logstr += 'nhidden: %d\n' % nhidden
    logstr += 'pooling: %s\n' % pool
    for lname, lyr in net.items():
        logstr += '%s %s\n' % (lname, str(get_output_shape(lyr, ASSUME)))
    logstr += '=========================== \n'
    print logstr
    log.write(logstr)
    log.flush()
    return net


def iterate_minibatches(inputs, targets, batchsize, rng=None, shuffle=False):
    ''' Taken from the mnist.py example of Lasagne'''
    # print inputs.shape, targets.size
    assert inputs.shape[0] == targets.size
    if shuffle:
        assert rng is not None
        indices = np.arange(inputs.shape[0])
        rng.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def learn_model(hyparams,
                train_path,
                val_path=None,
                test_path=None,
                vocab_file=None,
                log_path="",
                model_file=None,
                embeddings_file=None):
    '''
        Train to classify sentiment
        Returns the trained network
    '''
    RNG = np.random.RandomState(SEED)

    timestamp = time.strftime('%m%d%Y_%H%M%S')
    log_file = open(log_path + "/training_log_" + timestamp, "w+")

    print "Loading Dataset"
    train, dev, test, vmap = load_dataset(train_path, test_path, vocab_file, rng=RNG, devfile=val_path)
    y_train, X_train = train
    y_val, X_val = dev
    y_test, X_test = test

    # ### sanity check ###
    pad_char = u'♥'
    vmap[pad_char] = 0

    print "Training size", X_train.shape[0]
    print "Validation size", X_val.shape[0]
    print "Test size", X_test.shape[0]

    V = len(vmap)
    n_classes = len(set(y_train))
    print "Vocab size:", V
    print "Number of classes", n_classes
    print "Classes", set(y_train)

    log_file.write('ntrain: %d\nnval: %d\nntest: %d\nnclasses: %d\nvocab size: %d\nbatchsize: %d\n' %
                   (X_train.shape[0], X_val.shape[0], X_test.shape[0], n_classes, V, hyparams.batchsize))

    # Initialize theano variables for input and output
    X = T.imatrix('X')
    M = T.matrix('M')
    y = T.ivector('y')

    log_file.write(str(hyparams) + '\n')
    log_file.flush()

    # Construct network
    print "Building Model"
    network = build_model(hyparams, vmap, log_file, n_classes, invar=X, maskvar=M)

    if model_file is not None:
        read_model_data(network, model_file)
        log_file.write('loaded params from file: %s\n' % model_file)
        log_file.flush()

    # Get network output
    output = lasagne.layers.get_output(network['softmax'])

    # Define objective function (cost) to minimize, mean crossentropy error
    cost = lasagne.objectives.categorical_crossentropy(output, y).mean()

    # Compute gradient updates
    params = lasagne.layers.get_all_params(network.values())

    grad_updates = None
    optim = hyparams.optimizer
    if optim == 'adagrad':
        grad_updates = lasagne.updates.adagrad(cost, params, learning_rate=hyparams.learning_rate)
    elif optim == 'adadelta':
        grad_updates = lasagne.updates.adadelta(cost, params, learning_rate=hyparams.learning_rate)
    elif optim == 'adam':
        grad_updates = lasagne.updates.adam(cost, params)
    else:
        raise Exception('unsupported optimizer: %s' % optim)

    # Compile train objective
    print "Compiling training functions"
    train = theano.function([X, M, y], cost,
                            updates=grad_updates,
                            allow_input_downcast=True)

    # need to switch off droput while testing
    test_output = lasagne.layers.get_output(network['softmax'], deterministic=True)
    val_cost_fn = lasagne.objectives.categorical_crossentropy(test_output, y).mean()
    preds = T.argmax(test_output, axis=1)
    val_acc_fn = T.mean(T.eq(preds, y), dtype=theano.config.floatX)
    val_fn = theano.function([X, M, y], [val_cost_fn, val_acc_fn, preds], allow_input_downcast=True)

    csv = open('%s/data_%s' % (log_path, timestamp), 'w+')
    csv.write('epoch, nexamples, train_loss, val_loss, val_acc\n')
    csv.flush()

    batchsize = hyparams.batchsize
    num_epochs = hyparams.nepochs

    def compute_val_error(log_file=log_file, X_val=X_val, y_val=y_val):
        val_loss = 0.
        val_acc = 0.
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val,
                                         batchsize, shuffle=False):
            x_val_mini, y_val_mini = batch
            v_loss, v_acc, _ = val_fn(x_val_mini[:, :, 0],
                                      x_val_mini[:, :, 1],
                                      y_val_mini)
            val_loss += v_loss
            val_acc += v_acc
            val_batches += 1

        try:
            val_loss /= val_batches
            val_acc /= val_batches
            log_file.write("\t  validation loss:\t\t{:.6f}\n".format(val_loss))
            log_file.write("\t  validation accuracy:\t\t{:.2f} %\n".format(val_acc * 100.))
        except ZeroDivisionError:
            print('WARNING: val_batches == 0')

        return val_loss, val_acc

    print "Starting Training"
    nbatches = X_train.shape[0] / batchsize
    valfreq = max(int(nbatches / 32.), 2)  # evaluate every [valfreq] minibatches

    begin_time = time.time()
    best_val_acc = -np.inf
    for epoch in xrange(num_epochs):
        train_err = 0.
        train_batches = 0
        start_time = time.time()
        # if epoch > 5:
        #     learn_rate /= 2
        for batch in iterate_minibatches(X_train, y_train, batchsize, rng=RNG, shuffle=True):
            x_mini, y_mini = batch
            # print x_train.shape, y_train.shape
            train_err += train(x_mini[:, :, 0], x_mini[:, :, 1], y_mini)
            train_batches += 1
            print '[epoch %d batch %d/%d]' % (epoch, train_batches, nbatches)
            # print "Batch {} : cost {:.6f}".format(
            #     train_batches, train_err / train_batches)

            if train_batches % valfreq == 0:
                log_file.write("\tBatch {} of epoch {} took {:.3f}s\n".format(
                    train_batches, epoch+1, time.time() - start_time))
                log_file.write("\t  training loss:\t\t{:.6f}\n".format(
                    train_err / train_batches))

                val_loss, val_acc = compute_val_error(X_val=X_val, y_val=y_val)

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    print '\t%f' % best_val_acc
                    write_model_data(network, log_path + '/best_lstm_model')

                log_file.write(
                    "\tCurrent best validation accuracy:\t\t{:.2f}\n".format(
                        best_val_acc * 100.))
                log_file.flush()

                data = [epoch, train_batches*batchsize, train_err/train_batches, val_loss, val_acc]
                csv.write(', '.join([str(d) for d in data]) + '\n')
                csv.flush()

        disp_msg = "Epoch {} of {} took {:.3f}s\n".format(
            epoch + 1, num_epochs, time.time() - start_time)
        print disp_msg
        log_file.write(disp_msg)
        log_file.write("\t  training loss:\t\t{:.6f}\n".format(
            train_err / train_batches))
        val_loss, val_acc = compute_val_error(X_val=X_val, y_val=y_val)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            write_model_data(network, log_path + '/best_lstm_model')

        log_file.write("Current best validation accuracy:\t\t{:.2f}\n".format(
            best_val_acc * 100.))

        if (epoch) % 1 == 0:
            test_loss, test_acc, _ = val_fn(X_test[:, :, 0],
                                            X_test[:, :, 1], y_test)
            log_file.write("Test accuracy:\t\t{:.2f}\n".format(
                test_acc * 100.))

        log_file.flush()
        # print ("preds  0:", (val_err[2] == 0).sum(), "1:",
        #        (val_err[2] == 1).sum(), "2:", (val_err[2] == 2).sum()
        #        )
        # print ("truth  0:", (y_val == 0).sum(), "1:",
        #        (y_val == 1).sum(), "2:", (y_val == 2).sum()
        #        )
    log_file.write("Training took {:.3f}s\n".format(time.time() - begin_time))

    read_model_data(network, log_path + '/best_lstm_model')
    test_loss, test_acc, _ = val_fn(X_test[:, :, 0], X_test[:, :, 1], y_test)
    log_file.write("Best Model Test accuracy:\t\t{:.2f}%\n".format(
        test_acc * 100.))

    log_file.close()
    csv.close()

    return network


def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./', '%s.%s' % (filename, 'params'))
    with open(filename, 'r') as f:
        data = cPickle.load(f)
    lasagne.layers.set_all_param_values(model.values(), data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model.values())
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, 'params')
    with open(filename, 'w+') as f:
        cPickle.dump(data, f)


def pad_mask(X, pad_with=0, maxlen=MAXLEN):
    N = len(X)
    X_out = None
    if pad_with == 0:
        X_out = np.zeros((N, maxlen, 2), dtype=np.int32)
    else:
        X_out = np.ones((N, maxlen, 2), dtype=np.int32) * pad_with
    for i, x in enumerate(X):
        n = len(x)
        if n < maxlen:
            X_out[i, :n, 0] = x
            X_out[i, :n, 1] = 1
        else:
            X_out[i, :, 0] = x[:maxlen]
            X_out[i, :, 1] = 1
    return X_out


def load_dataset(trainfile, testfile, vocabfile, devfile=None, rng=None, pad_with=0):
    def load_file(fname, pad_with=0):
        X, Y = [], []
        nerrs = 0
        with open(fname, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    nerrs += 1
                    continue
                y, x = parts[0], parts[1]
                if len(x) == 0:
                    nerrs += 1
                    continue
                y = int(y)
                x = map(int, x.split(' '))
                #  shift up by 1 so padding doesnt perturb input
                x = map(lambda i: i + 1, x)
                Y.append(y)
                X.append(x)
        print 'bad lines: ', nerrs
        return np.asarray(Y, dtype=np.int32), pad_mask(X, pad_with=pad_with)
    vocab = cPickle.load(open(vocabfile, 'r'))
    vocab_shift = {}
    V = len(vocab)
    for k, v in vocab.items():
        vocab_shift[k] = vocab[k] + 1
    pad_with = 0
    trainy, trainx = load_file(trainfile, pad_with=pad_with)
    testy, testx = load_file(testfile, pad_with=pad_with)
    if devfile:
        devy, devx = load_file(devfile, pad_with=pad_with)
    else:
        assert rng is not None
        n = len(trainy)
        nval = int(0.2 * n)
        indices = np.arange(n)
        rng.shuffle(indices)
        train_indices = indices[:n-nval]
        val_indices = indices[n-nval:]
        devx = trainx[val_indices]
        devy = trainy[val_indices]
        trainx = trainx[train_indices]
        trainy = trainy[train_indices]
    train = (trainy, trainx)
    dev = (devy, devx)
    test = (testy, testx)
    return train, dev, test, vocab_shift


def test_model(model_path,
               train_path,
               val_path=None,
               test_path=None,
               vocab_file=None,
               label_file=None,
               output_file=None,
               embeddings_file=None):

    RNG = np.random.RandomState(SEED)
    timestamp = time.strftime('%m%d%Y_%H%M%S')
    assert output_file is not None
    log_file = open(output_file, 'w+')
    log_file.write('%s\n' % timestamp)

    print "Loading Dataset"
    _, dev, test, vmap = load_dataset(train_path, test_path, vocab_file, rng=RNG, devfile=val_path)
    y_val, X_val = dev
    y_test, X_test = test
    pad_char = u'♥'
    vmap[pad_char] = 0

    log_file.write('dev: %d, test: %d\n' % (X_val.shape[0], X_test.shape[0]))

    classes = cPickle.load(open(label_file, 'r'))
    classnames = list(map(lambda x: x[0], sorted(classes.items(), key=lambda x: x[1])))
    n_classes = len(classnames)
    log_file.write('classes: %s\n' % str(classes))

    # Initialize theano variables for input and output
    X = T.imatrix('X')
    M = T.matrix('M')
    y = T.ivector('y')

    # Construct network
    print "Building Model"
    log_file.write('model file: %s\n' % model_path)
    #    network = build_model(hyparams, vmap, log_file, n_classes, invar=X, maskvar=M)

    network = build_model(hyparams, vmap, log_file, n_classes, invar=X, maskvar=M)
    read_model_data(network, model_path)

    # need to switch off droput while testing
    test_output = lasagne.layers.get_output(network['softmax'], deterministic=True)
    val_cost_fn = lasagne.objectives.categorical_crossentropy(test_output, y).mean()
    preds = T.argmax(test_output, axis=1)
    val_acc_fn = T.mean(T.eq(preds, y), dtype=theano.config.floatX)
    val_fn = theano.function([X, M, y], [val_cost_fn, val_acc_fn, preds], allow_input_downcast=True)

    val_loss, val_acc, val_pred = val_fn(X_val[:, :, 0], X_val[:, :, 1], y_val)
    dev_eval = ConfusionMatrix(y_val, val_pred, classnames)
    log_file.write('DEV RESULTS\n')
    log_file.write('%s\n' % str(dev_eval))

    test_loss, test_acc, test_pred = val_fn(X_test[:, :, 0], X_test[:, :, 1], y_test)
    test_eval = ConfusionMatrix(y_test, test_pred, classnames)
    log_file.write('TEST RESULTS\n')
    log_file.write('%s\n' % str(test_eval))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='train word-level lstm')
    # input
    p.add_argument('--tweet-file', required=True, help='path to train data')
    p.add_argument('--test-file', help='path to test file')
    p.add_argument('--dev-file', type=str, help='path to dev set')
    p.add_argument('--vocab', required=True, help='path to vocabulary')
    p.add_argument('--model-file', type=str)
    p.add_argument('--label-file', type=str)

    # output
    p.add_argument('--log-path', type=str, required=True, help='path to store log file')
    p.add_argument('--results-file', type=str, help='filename for results file')

    # hyperparameters
    p.add_argument('--nepochs', type=int, default=30, help='# of epochs')
    p.add_argument('--batchsize', type=int, default=512, help='batch size')
    p.add_argument('--learning-rate', type=float, default=0.1, help='learning rate')
    p.add_argument('--bidirectional', type=int, default=1, help='bidirectional LSTM?')
    p.add_argument('--nhidden', type=int, default=256, help='num hidden units')
    p.add_argument('--embedding-dim', type=int, default=50, help='embedding size')
    p.add_argument('--pool', type=str, default='mean', help='pooling strategy')
    p.add_argument('--grad-clip', type=int, default=100, help='gradient clipping')
    p.add_argument('--optimizer', type=str, default='adam', help='optimizer')

    # switches
    p.add_argument('--test-only', type=int, default=0, help='just test model contained in model file')

    args = p.parse_args()
    print("ARGS:")
    print(args)
    hyparams = HParams()
    hyparams.parse_args(args)
    print hyparams

    if args.test_only:
        test_model(args.model_file,
                   train_path=args.tweet_file,
                   vocab_file=args.vocab,
                   test_path=args.test_file,
                   output_file=args.results_file,
                   label_file=args.label_file
                   )
    else:
        learn_model(hyparams,
                    args.tweet_file,
                    vocab_file=args.vocab,
                    test_path=args.test_file,
                    val_path=args.dev_file,
                    log_path=args.log_path,
                    model_file=args.model_file
                    )


