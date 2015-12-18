# -*- coding: utf-8 -*-
import cPickle
import os
import time
import argparse
import codecs

import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as layer
from lasagne.layers import get_output_shape


MAXLEN = 140


def build_model(vmap,
                nclasses=2,
                embedding_dim=50,
                nhidden=256,
                batchsize=None,
                invar=None,
                maskvar=None,
                bidirectional=True,
                pool=True,
                grad_clip=100,
                maxlen=MAXLEN):

    V = len(vmap)
    W = lasagne.init.Normal()

    # Input Layer
    # TODO: should be (batchsize, maxlen, vocab_size)
    l_in = layer.InputLayer((batchsize, maxlen, V), input_var=invar)
    l_mask = layer.InputLayer((batchsize, maxlen), input_var=maskvar)
    ASSUME = {l_in: (200, 140, 94), l_mask: (200, 140)}
    print 'Input Layer'
    print 'output:', get_output_shape(l_in, ASSUME)
    print 'output(mask):', get_output_shape(l_mask, ASSUME)
    print

    # Embedding Layer
    l_emb = layer.EmbeddingLayer(l_in, input_size=V, output_size=embedding_dim, W=W)
    print 'Embedding Layer'
    print 'output:', get_output_shape(l_emb, ASSUME)

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

    l_fwd = layer.LSTMLayer(
        l_emb,
        num_units=nhidden,
        grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh,
        mask_input=l_mask,
        ingate=gate_params,
        forgetgate=gate_params,
        cell=cell_params,
        outgate=gate_params,
        learn_init=True
    )

    print 'Forward LSTM'
    print 'output:', get_output_shape(l_fwd, ASSUME)

    l_concat = None
    if bidirectional:
        l_bwd = layer.LSTMLayer(
            l_emb,
            num_units=nhidden,
            grad_clipping=grad_clip,
            nonlinearity=lasagne.nonlinearities.tanh,
            mask_input=l_mask,
            ingate=gate_params,
            forgetgate=gate_params,
            cell=cell_params,
            outgate=gate_params,
            learn_init=True,
            backwards=True
        )
        print 'Backward LSTM'
        print 'output:', get_output_shape(l_bwd, ASSUME)

        def tmean(a, b):
            agg = theano.tensor.add(a, b)
            agg /= 2.
            return agg

        if pool:
            l_concat = layer.ElemwiseMergeLayer([l_fwd, l_bwd], tmean)
        else:
            l_concat = layer.ConcatLayer([l_fwd, l_bwd])
    else:
        l_concat = layer.ConcatLayer([l_fwd])
    print 'Concat'
    print 'output:', get_output_shape(l_concat, ASSUME)

    l_concat = layer.DropoutLayer(l_concat, p=0.5)

    l_lstm2 = layer.LSTMLayer(
        l_concat,
        num_units=nhidden,
        grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh,
        mask_input=l_mask,
        ingate=gate_params,
        forgetgate=gate_params,
        cell=cell_params,
        outgate=gate_params,
        learn_init=True,
        only_return_final=True
    )

    print 'LSTM #2'
    print 'output:', get_output_shape(l_lstm2, ASSUME)

    l_lstm2 = layer.DropoutLayer(l_lstm2, p=0.6)

    network = layer.DenseLayer(
        l_lstm2,
        num_units=nclasses,
        nonlinearity=lasagne.nonlinearities.softmax
    )

    print 'Dense Layer'
    print 'output:', get_output_shape(network, ASSUME)

    return network


def learn_model(train_path,
                val_path=None,
                test_path=None,
                num_epochs=5,
                batchsize=64,
                learn_rate=0.1,
                vocab_file=None,
                val_ratio=0.1,
                log_path=""):
    '''
        Train to classify sentiment
        Returns the trained network
    '''
    train, dev, test, vmap = load_dataset(train_path, test_path, vocab_file, devfile=val_path)
    trainy, trainx = train
    devy, devx  = dev
    testy, testx = test

    V = len(vmap)
    nclasses = len(set(trainy))

    print "# training examples", trainy.shape[0]
    print "# validation examples", devy.shape[0]
    print "# test examples", testy.shape[0]
    print "example shape", trainx[0].shape
    print "vocab size", V
    print "# classes", nclasses

    check = trainx[0]
    print "check:", check.shape
    print
    print "input:", check[:, :, 0].shape
    print check[:, :, 0]
    print
    print "mask:", check[:, :, 1].shape
    print check[:, :, 1]

    X = T.itensor3('X')
#    M = T.tensor3('M')
    M = T.matrix('M')
    y = T.ivector('y')

    print "building model"
    network = build_model(vmap, nclasses, invar=X, maskvar=M)
    output = layer.get_output(network)
    cost = lasagne.objectives.categorical_crossentropy(output, y).mean()
    params = layer.get_all_params(network)
    updates = lasagne.updates.adam(cost, params)

    test_output = layer.get_output(network, deterministic=True)

    val_cost = lasagne.objectives.categorical_crossentropy(test_output, y).mean()
    predictions = T.argmax(test_output, axis=1)
    val_accuracy = T.mean(T.eq(predictions, y), dtype=theano.config.floatX)

    print "compiling functions"
    trainfxn = theano.function([X, M, y], cost, updates=updates, allow_input_downcast=True)
    valfxn = theano.function([X, M, y], [val_cost, val_accuracy, predictions], allow_input_downcast=True)

    def compute_validation_error(log, x, y, batch_size):
        loss = acc = 0.
        batches = 0
        for batch in iterate_minibatches(x, y, batch_size, shuffle=False):
            xmini, ymini = batch
            bloss, bacc, _ = valfxn(xmini[:,:,:,0], xmini[:,:,0,1], ymini)
            loss += bloss
            acc += bacc
            batches += 1
        try:
            loss /= batches
            acc /= batches
            log.write('\tvalidation loss:\t\t{:.6f}\n'.format(loss))
            log.write('\tvalidation acc:\t\t{:.2f}\n'.format(acc))
        except ZeroDivisionError:
            print('warning: %d validation batches' % batches)
        return loss, acc

    logfilename = '%s/training_log_%s' % (log_path, time.strftime('%m%d%Y_%H%M%S'))
    logfile = open(logfilename, 'w+')

    print "starting training"
    start = time.time()
    best_val_acc = -np.inf
    for epoch in xrange(num_epochs):
        train_err = 0.
        train_batches = 0
        epoch_start = time.time()
        for batch in iterate_minibatches(trainx, trainy, batchsize, shuffle=True):
            xmini, ymini = batch
            train_err += trainfxn(xmini[:,:,:,0], xmini[:,:,0,1], ymini)
            train_batches += 1
            if (train_batches % 512) == 0:
                logfile.write('\tbatch {} of epoch {} took {:3f}s\n'.format(train_batches, epoch, time.time() - epoch_start))
                logfile.write('\t\ttraining loss: {:.6f}\n'.format(train_err / train_batches))
                val_loss, val_acc = compute_validation_error(logfile, devx, devy, batchsize)
                print('epoch %d, batch %d: val. loss %.3f, val. acc %.3f' % (epoch, train_batches, val_loss, val_acc))
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    write_model_data(network, log_path + '/best_lstm_model')
                logfile.write('current best validation accuracy: {:.2f}\n'.format(best_val_acc * 100.))
                logfile.flush()
        disp = 'epoch {} / {} took {:.3f}s\n'.format(epoch, num_epochs, time.time() - epoch_start)
        print disp
        logfile.write(disp)
        logfile.write('\ttraining loss: {:.6f}\n'.format(train_err / train_batches))
        val_loss, val_acc = compute_validation_error(logfile, devx, devy, batchsize)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            write_model_data(network, log_path + '/best_lstm_model')
        logfile.write('current best validation accuracy: {:.2f}\n'.format(best_val_acc * 100.))
        if (epoch % 1) == 0:
            test_loss, test_acc, _ = valfxn(testx[:,:,:,0], testx[:,:,0,1], testy)
            logfile.write('test accuracy: {:.2f}\n'.format(test_acc * 100.))
        logfile.flush()
    logfile.write("Training took {:.3f}s\n".format(time.time() - start))
    network = read_model_data(network, log_path + '/best_lstm_model')
    test_loss, test_acc, _ = valfxn(testx[:,:,:,0], testx[:,:,0,1])
    logfile.write("Best Model Test accuracy:\t\t{:.2f}%\n".format(test_acc * 100.))
    logfile.close()
    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    ''' Taken from the mnist.py example of Lasagne'''
    # print inputs.shape, targets.size
    assert inputs.shape[0] == targets.size
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def load_dataset(trainfile, testfile, vocabfile, devfile=None):
    def pad_mask(X, vocab_size, maxlen=MAXLEN):
        N = len(X) # num examples
        U = np.zeros((N, maxlen, vocab_size, 2))
        # U_mask = np.zeros((N, maxlen, vocab_size))
        for i in xrange(N):
            x = X[i]
            n = len(x) # no. chars
            xh = np.zeros((maxlen, vocab_size))
            end = min(n, maxlen)
            for j, c in enumerate(x[:end]):
                xh[j, c] = 1
            U[i, :, :, 0] = xh
            U[i, :, 0, 1] = 1
            # if n < maxlen:
            #     U[i, :n, :, 0] = xh
            #     U[i, :n, :, 1] = 1
            # else:
            #     U[i, :maxlen, :, 0] = xh
            #     U[i, :maxlen, :, 1] = 1
            # xh = np.zeros((maxlen, vocab_size))
            # xh_mask = np.zeros(maxlen)
            # end = min(n, maxlen)
            # for j, c in enumerate(x[:end]):
            #     xh[j, c] = 1
            #     xh_mask[j] = 1
            # U[i, :, :, 0] = xh
            # U[i, :, :, 1] = xh_mask
        print "U:", U.shape
        return U

    def load_file(fname, vocab_size):
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
        Y = np.asarray(Y, dtype=np.int32)
        X = pad_mask(X, vocab_size)
        return Y, X

    vocab = cPickle.load(open(vocabfile, 'r'))
    vocab_shift = {}
    for k, v in vocab.items():
        vocab_shift[k] = vocab[k] + 1
    pad_char = u'♥"'
    vocab_shift[pad_char] = 0
    V = len(vocab_shift)
    trainy, trainx = load_file(trainfile, V)
    testy, testx = load_file(testfile, V)
    if devfile:
        devy, devx = load_file(devfile, V)
    else:
        n = len(trainy)
        nval = int(0.2 * n)
        indices = np.arange(n)
        np.random.shuffle(indices)
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


def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./', '%s.%s' % (filename, 'params'))
    with open(filename, 'r') as f:
        data = cPickle.load(f)
    lasagne.layers.set_all_param_values(model, data)
    return model


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, 'params')
    with open(filename, 'w+') as f:
        cPickle.dump(data, f)


def test_model(args):
    _, dev, test, vmap = load_dataset(args.tweet_file, args.testfile, args.vocab)
    labelmap = cPickle.load(open(args.label_file, 'r'))
    nclasses = len(labelmap)

    X = T.itensor3('X')
    M = T.matrix('M')
    y = T.ivector('y')

    print "building model"
    network = build_model(vmap, nclasses, invar=X, maskvar=M)
    print "loading params"
    network = read_model_data(network, args.model_file)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='train word-level lstm')
    p.add_argument('--tweet-file', required=True, help='path to train data')
    p.add_argument('--vocab', required=True, help='path to vocabulary')
    p.add_argument('--label-file', type=str)
    p.add_argument('--log-path', type=str, required=True, help='path to store log file')
    p.add_argument('--model-file', type=str)

    p.add_argument('--test-file', help='path to test file')

    p.add_argument('--nepochs', type=int, default=30, help='# of epochs')
    p.add_argument('--batchsize', type=int, default=512, help='batch size')
    p.add_argument('--learning-rate', type=float, default=0.1, help='learning rate')

    args = p.parse_args()
    print("ARGS:")
    print(args)

    learn_model(
        train_path=args.tweet_file,
        vocab_file=args.vocab,
        test_path=args.test_file,
        num_epochs=args.nepochs,
        batchsize=args.batchsize,
        learn_rate=args.learning_rate,
        log_path=args.log_path
    )