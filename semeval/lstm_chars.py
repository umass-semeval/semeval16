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
                bidirectional=False,
                pool=False,
                grad_clip=100,
                maxlen=MAXLEN):

    V = len(vmap)
    K = nclasses
    W = lasagne.init.Normal()

    # Input Layer
    l_in = layer.InputLayer((batchsize, maxlen), input_var=invar)
    l_mask = layer.InputLayer((batchsize, maxlen), input_var=maskvar)
    ASSUME = {l_in: (200, 140), l_mask: (200, 140)}

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

    l_fwd = layer.DropoutLayer(l_fwd, p=0.5)

    l_lstm2 = layer.LSTMLayer(
        l_fwd,
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

    # # Mean Pooling Layer
    # pool_size = 16
    # l_pool = layer.FeaturePoolLayer(l_lstm2, pool_size)
    # print('Mean Pool Layer Shape:')
    # print 'output:', get_output_shape(l_pool, ASSUME)
    # print

    network = layer.DenseLayer(
        l_lstm2,
        num_units=nclasses,
        nonlinearity=lasagne.nonlinearities.softmax
    )

    print 'Dense Layer'
    print 'output:', get_output_shape(network, ASSUME)

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


def learn_model(train_path,
                val_path=None,
                test_path=None,
                max_norm=5,
                num_epochs=5,
                batchsize=64,
                learn_rate=0.1,
                vocab_file=None,
                val_ratio=0.1,
                log_path="",
                embeddings_file=None):
    '''
        Train to classify sentiment
        Returns the trained network
    '''

    print "Loading Dataset"
    # X_train, y_train, X_val, y_val, X_test, vmap = load_dataset(train_path,
    #                                                             val_path,
    #                                                             test_path)

    train, dev, test, vmap = load_dataset(train_path, test_path, vocab_file, devfile=val_path)
    y_train, X_train = train
    y_val, X_val = dev
    y_test, X_test = test
    # X_train, y_train, X_val, y_val, X_test, y_test, vmap = load_big_dataset(
    #     train_path, test_path, vocab_file, val_ratio)

    print "Training size", X_train.shape[0]
    print "Validation size", X_val.shape[0]
    print "Test size", X_test.shape[0]
    # print X_train.shape, len(y_train)
    # print X_train[0], y_train[0]
    V = len(vmap)
    n_classes = len(set(y_train))
    print "Vocab size:", V
    print "Number of classes", n_classes
    print "Classes", set(y_train)

    # Initialize theano variables for input and output
    X = T.imatrix('X')
    M = T.matrix('M')
    y = T.ivector('y')

    # Construct network
    print "Building Model"
    network = build_model(vmap, n_classes, invar=X, maskvar=M)

    # network = None
    # if embeddings_file:
    #     network = build_model(vmap, n_classes, invar=X, maskvar=M, ini_word2vec=True, word2vec_file=embeddings_file)
    # else:
    #     network = build_model(vmap, n_classes, input_var=X, mask_var=M)

    # Get network output
    output = lasagne.layers.get_output(network)

    # Define objective function (cost) to minimize, mean crossentropy error
    cost = lasagne.objectives.categorical_crossentropy(output, y).mean()

    # Compute gradient updates
    params = lasagne.layers.get_all_params(network)
    # grad_updates = lasagne.updates.nesterov_momentum(cost, params,learn_rate)
    grad_updates = lasagne.updates.adam(cost, params)
    # grad_updates = lasagne.updates.adadelta(cost, params, learn_rate)

    # Compile train objective
    print "Compiling training functions"
    train = theano.function([X, M, y], cost,
                            updates=grad_updates,
                            allow_input_downcast=True)

    # need to switch off droput while testing
    test_output = lasagne.layers.get_output(network, deterministic=True)
    val_cost_fn = lasagne.objectives.categorical_crossentropy(
        test_output, y).mean()
    preds = T.argmax(test_output, axis=1)
    val_acc_fn = T.mean(T.eq(preds, y),
                        dtype=theano.config.floatX)
    val_fn = theano.function([X, M, y], [val_cost_fn, val_acc_fn, preds],
                             allow_input_downcast=True)

    log_file = open(log_path + "/training_log_" +
                    time.strftime('%m%d%Y_%H%M%S'), "w+")

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
    begin_time = time.time()
    best_val_acc = -np.inf
    for epoch in xrange(num_epochs):
        train_err = 0.
        train_batches = 0
        start_time = time.time()
        # if epoch > 5:
        #     learn_rate /= 2
        for batch in iterate_minibatches(X_train, y_train,
                                         batchsize, shuffle=True):
            x_mini, y_mini = batch
            # print x_train.shape, y_train.shape
            train_err += train(x_mini[:, :, 0], x_mini[:, :, 1], y_mini)
            train_batches += 1
            # print "Batch {} : cost {:.6f}".format(
            #     train_batches, train_err / train_batches)

            if train_batches % 512 == 0:
                log_file.write("\tBatch {} of epoch {} took {:.3f}s\n".format(
                    train_batches, epoch+1, time.time() - start_time))
                log_file.write("\t  training loss:\t\t{:.6f}\n".format(
                    train_err / train_batches))

                val_loss, val_acc = compute_val_error(X_val=X_val, y_val=y_val)

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    write_model_data(network, log_path + '/best_lstm_model')

                log_file.write(
                    "\tCurrent best validation accuracy:\t\t{:.2f}\n".format(
                        best_val_acc * 100.))
                log_file.flush()

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

    network = read_model_data(network, log_path + '/best_lstm_model')
    test_loss, test_acc, _ = val_fn(X_test[:, :, 0], X_test[:, :, 1], y_test)
    log_file.write("Best Model Test accuracy:\t\t{:.2f}%\n".format(
        test_acc * 100.))

    log_file.close()

    return network


def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./', '%s.%s' % (filename, 'params'))
    with open(filename, 'r') as f:
        data = cPickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, 'params')
    with open(filename, 'w+') as f:
        cPickle.dump(data, f)

def pad_mask(X, maxlen=MAXLEN):
    N = len(X)
    X_out = np.zeros((N, maxlen, 2), dtype=np.int32)
    for i, x in enumerate(X):
        n = len(x)
        if n < maxlen:
            X_out[i, :n, 0] = x
            X_out[i, :n, 1] = 1
        else:
            X_out[i, :, 0] = x[:maxlen]
            X_out[i, :, 1] = 1
    return X_out


def load_dataset(trainfile, testfile, vocabfile, devfile=None):
    def load_file(fname):
        X, Y = [], []
        nerrs = 0
        with open(fname, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    nerrs += 1
                    continue
                y, x = parts[0], parts[1]
                y = int(y)
                x = map(int, x.split(' '))
                Y.append(y)
                X.append(x)
        print 'bad lines: ', nerrs
        return np.asarray(Y, dtype=np.int32), pad_mask(X)
    vocab = cPickle.load(open(vocabfile, 'r'))
    trainy, trainx = load_file(trainfile)
    testy, testx = load_file(testfile)
    if devfile:
        devy, devx = load_file(devfile)
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
    return train, dev, test, vocab


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='train word-level lstm')
    p.add_argument('--tweet-file', required=True, help='path to train data')
    p.add_argument('--vocab', required=True, help='path to vocabulary')
    p.add_argument('--log-path', type=str, required=True, help='path to store log file')

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

    # root = '/home/kate/F15/semeval16/chars'
    # vocab = '%s/%s' % (root, 'train.chars.tsv.vocab.pkl')
    # trainf = '%s/%s' % (root, 'train.chars.tsv')
    # devf = '%s/%s' % (root, 'dev.chars.tsv')
    # testf = '%s/%s' % (root, 'test.chars.tsv')
    #
    # learn_model(
    #     train_path=trainf,
    #     val_path=devf,
    #     test_path=testf,
    #     vocab_file=vocab,
    #     log_path='%s/char_lstm_result' % root
    # )
    #
    # # train, dev, test, V = load_semeval_dataset(trainf, devf, testf, vocab)


