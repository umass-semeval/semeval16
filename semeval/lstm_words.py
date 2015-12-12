import numpy as np
import theano
import theano.tensor as T

import time
import Tweet

import lasagne
from lasagne.layers import get_output_shape
from gensim.models import word2vec

import os
import pickle
import argparse


MAX_SEQ = 140  # maximum length of a sequence


def build_model(vmap,  # input vocab mapping
                num_classes,  # number classes to predict
                K=300,  # dimensionality of embeddings
                num_hidden=256,  # number of hidden_units
                batchsize=None,  # size of each batch (None for variable size)
                input_var=None,  # theano variable for input
                mask_var=None,  # theano variable for input mask
                bidirectional=True,  # whether to use bi-directional LSTM
                mean_pooling=True,
                grad_clip=100.,  # gradients above this will be clipped
                max_seq_len=MAX_SEQ,  # maximum lenght of a sequence
                ini_word2vec=False,  # whether to initialize with word2vec
                word2vec_file='/iesl/canvas/tbansal/glove.twitter.27B.200d.txt',
                # location of trained word vectors
                ):

    V = len(vmap)
    # basic input layer (batchsize, SEQ_LENGTH),
    # None lets us use variable bs
    # use a mask to outline the actual input sequence
    if ini_word2vec:
        print('loading embeddings from file %s' % word2vec_file)
        word2vec_model = word2vec.Word2Vec.load_word2vec_format(word2vec_file, binary=False)
        print 'done.'
        K = word2vec_model[word2vec_model.vocab.keys()[0]].size  # override dim
        print('embedding dim: %d' % K)
        W = np.zeros((V, K), dtype=np.float32)
        no_vectors = 0
        for w in vmap:
            if w in word2vec_model.vocab:
                W[vmap[w]] = np.asarray(word2vec_model[w], dtype=np.float32)
            else:
                W[vmap[w]] = np.random.normal(scale=0.01, size=K)
                no_vectors += 1
        W = theano.shared(W)
        print " Initialized with word2vec. Couldn't find", no_vectors, "words!"
    else:
        W = lasagne.init.Normal()

    # Input Layer
    l_in = lasagne.layers.InputLayer((batchsize, max_seq_len), input_var=input_var)
    l_mask = lasagne.layers.InputLayer((batchsize, max_seq_len), input_var=mask_var)

    HYPOTHETICALLY = {l_in: (200, 140), l_mask: (200, 140)}

    print('Input Layer Shape:')
    LIN = get_output_shape(l_in, HYPOTHETICALLY)
    print 'input:', HYPOTHETICALLY
    print 'output:', LIN
    print

    # Embedding layer
    l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=V, output_size=K, W=W)
    print('Embedding Layer Shape:')
    print 'input:', LIN
    print 'output:', get_output_shape(l_emb, HYPOTHETICALLY)
    print

    # add droput
    # l_emb = lasagne.layers.DropoutLayer(l_emb, p=0.2)

    # Use orthogonal Initialization for LSTM gates
    gate_params = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.)
    )
    cell_params = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        W_cell=None, b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.tanh
    )

    l_fwd = lasagne.layers.LSTMLayer(
        l_emb, num_units=num_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh, mask_input=l_mask,
        ingate=gate_params, forgetgate=gate_params, cell=cell_params,
        outgate=gate_params, learn_init=True
    )

    print('Forward LSTM Shape:')
    print 'input:', get_output_shape(l_emb, HYPOTHETICALLY)
    print 'output:', get_output_shape(l_fwd, HYPOTHETICALLY)
    print

    # add droput
    # l_fwd = lasagne.layers.DropoutLayer(l_fwd, p=0.5)

    if bidirectional:
        # add a backwards LSTM layer for bi-directional
        l_bwd = lasagne.layers.LSTMLayer(
            l_emb, num_units=num_hidden, grad_clipping=grad_clip,
            nonlinearity=lasagne.nonlinearities.tanh, mask_input=l_mask,
            ingate=gate_params, forgetgate=gate_params, cell=cell_params,
            outgate=gate_params, learn_init=True,
            backwards=True
        )
        print('Backward LSTM Shape:')
        print 'input:', get_output_shape(l_emb, HYPOTHETICALLY)
        print 'output:', get_output_shape(l_bwd, HYPOTHETICALLY)
        print

        # print "backward layer:", lasagne.layers.get_output_shape(
        #     l_bwd, {l_in: (200, 140), l_mask: (200, 140)})

        # concatenate forward and backward LSTM
        l_concat = lasagne.layers.ConcatLayer([l_fwd, l_bwd])
        print('Concat Layer Shape:')
        print 'input:', get_output_shape(l_fwd, HYPOTHETICALLY), get_output_shape(l_bwd, HYPOTHETICALLY)
        print 'output:', get_output_shape(l_concat, HYPOTHETICALLY)
        print
    else:
        l_concat = l_fwd
        print('Concat Layer Shape:')
        print 'input:', get_output_shape(l_fwd, HYPOTHETICALLY)
        print 'output:', get_output_shape(l_concat, HYPOTHETICALLY)
        print

    # add droput
    l_concat = lasagne.layers.DropoutLayer(l_concat, p=0.5)

    l_lstm2 = lasagne.layers.LSTMLayer(
        l_concat,
        num_units=num_hidden,
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

    print('LSTM Layer #2 Shape:')
    print 'input:', get_output_shape(l_concat, HYPOTHETICALLY)
    print 'output:', get_output_shape(l_lstm2, HYPOTHETICALLY)
    print

    # add dropout
    l_lstm2 = lasagne.layers.DropoutLayer(l_lstm2, p=0.6)

    # Mean Pooling Layer
    pool_size = 16
    l_pool = lasagne.layers.FeaturePoolLayer(l_lstm2, pool_size)
    print('Mean Pool Layer Shape:')
    print 'input:', get_output_shape(l_lstm2, HYPOTHETICALLY)
    print 'output:', get_output_shape(l_pool, HYPOTHETICALLY)
    print

    # Dense Layer
    network = lasagne.layers.DenseLayer(
        l_pool,
        num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    print('Dense Layer Shape:')
    print 'input:', get_output_shape(l_pool, HYPOTHETICALLY)
    print 'output:', get_output_shape(network, HYPOTHETICALLY)

    return network


def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./', '%s.%s' % (filename, 'params'))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, 'params')
    with open(filename, 'w+') as f:
        pickle.dump(data, f)


def preprocess(tweets, vmap=None, stopf='../lexica/stopwords.txt'):
    ''' Code to clean, remove stopwords and tokenize the tweet '''
    if vmap is None:
        with open(stopf, mode='r') as f:
            stopwords = map(lambda x: x.strip(), f.readlines())
        V = 0
        vmap = {}
        ntweets = len(tweets)
        indices = np.arange(ntweets)
        np.random.shuffle(indices)
        for id in indices:
            tweet = tweets[id]
            for w in tweet.raw_text.strip().split():
                w = w.lower()
                if w not in vmap and w not in stopwords:
                    vmap[w] = V
                    V += 1

    X = []
    y = []
    label_map = {'negative': 0,
                 'neutral': 1,
                 'positive': 2}
    for tweet in tweets:
        txt = tweet.raw_text
        out = []
        for w in txt.strip().split():
            w = w.lower()
            if w.strip() in vmap:
                out.append(vmap[w.strip()])
        if out:
            X.append(out)
            y.append(label_map[tweet.label])

    return X, y, vmap


def pad_mask(X, max_seq_length=MAX_SEQ):
    # last dim for mask

    N = len(X)
    X_out = np.zeros((N, max_seq_length, 2), dtype=np.int32)
    for i, x in enumerate(X):
        n = len(x)
        if n < max_seq_length:
            X_out[i, :n, 0] = x
            X_out[i, :n, 1] = 1
        else:
            X_out[i, :, 0] = x[:max_seq_length]
            X_out[i, :, 1] = 1

    return X_out


def load_dataset(train_file, val_file, test_file):
    ''' Load the semeval twitter data
        Use Kate's Tweet class
    '''

    class arbit:
        super
    args = arbit()

    args.subtask_id = 'a'
    args.train_file = train_file
    args.dev_file = val_file
    args.test_file = test_file

    train, val, test = Tweet.load_datasets(args)

    X_train, y_train, vmap = preprocess(train)
    X_val, y_val, _ = preprocess(val, vmap)
    X_test, y_test, _ = preprocess(test, vmap)

    X_train, X_val, X_test = map(pad_mask, [X_train, X_val, X_test])
    y_train, y_val, y_test = map(np.asarray, [y_train, y_val, y_test])

    return X_train, y_train, X_val, y_val, X_test, vmap


def load_big_dataset(tweet_file, test_file, vocab_file, val_ratio=0.05):
    '''
        Load the big 1.6 million Dataset
        tweet_file is location of the processed tweets
        val_ratio is the fraction of tweets required for validation
    '''
    vmap = {}
    with open(vocab_file, "r") as vf:
        for line in vf:
            # if len(line.split("\t")) > 3:
            #     id, _, _, cnt = line.strip().decode('utf-8').split("\t")
            #     w = r"\t"
            # else:
            id, w, cnt = line.strip().decode('utf-8').split("\t")
            vmap[w] = int(id)

    label_map = {'negative': 0,
                 'positive': 1}

    def read_data(infile):
        X = []
        y = []
        nerrs = 0
        with open(infile, "r") as tf:
            for line in tf:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    nerrs += 1
                    continue
                _, label, tweet = parts[0], parts[1], parts[2]
                if label == "neutral":
                    continue
                X.append(map(int, tweet.strip().split()))
                y.append(label_map[label.strip()])
        print('%d bad lines for file %s' % (nerrs, infile))
        X = pad_mask(X)
        y = np.asarray(y, dtype=np.int32)
        return X, y

    X, y = read_data(tweet_file)
    n = X.shape[0]
    n_val = int(val_ratio * n)
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_indices = indices[:n-n_val]
    val_indices = indices[n-n_val:]

    X_test, y_test = read_data(test_file)

    return (X[train_indices], y[train_indices],
            X[val_indices], y[val_indices],
            X_test, y_test,
            vmap)


def load_semeval_dataset(tweet_file, test_file, dev_file, vocab_file):
    '''
        Load the big 1.6 million Dataset
        tweet_file is location of the processed tweets
        val_ratio is the fraction of tweets required for validation
    '''
    vmap = {}
    with open(vocab_file, "r") as vf:
        for line in vf:
            # if len(line.split("\t")) > 3:
            #     id, _, _, cnt = line.strip().decode('utf-8').split("\t")
            #     w = r"\t"
            # else:
            id, w, cnt = line.strip().decode('utf-8').split("\t")
            vmap[w] = int(id)

    label_map = {'negative': 0,
                 'positive': 1}

    def read_data(infile):
        data = Tweet.load_from_tsv(infile)
        X = []
        y = []
        for tweet in data:
            if tweet.label == 'neutral':
                continue
            text = tweet.raw_text.lower()
            ints = []
            for w in text.split(' '):
                if w in vmap:
                    ints.append(vmap[w])
            lv = label_map[tweet.label]
            X.append(ints)
            y.append(lv)
        X = pad_mask(X)
        y = np.asarray(y, dtype=np.int32)
        return X, y

    trainx, trainy = read_data(tweet_file)
    devx, devy = read_data(dev_file)
    testx, testy = read_data(test_file)
    return trainx, trainy, devx, devy, testx, testy, vmap


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
                embeddings_file=None,
                params_file=None,
                use_semeval=False):
    '''
        Train to classify sentiment
        Returns the trained network
    '''

    print "Loading Dataset"
    if use_semeval:
         X_train, y_train, X_val, y_val, X_test, y_test, vmap = load_semeval_dataset(train_path, test_path, val_path, vocab_file)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, vmap = load_big_dataset(train_path, test_path, vocab_file, val_ratio)

    print "Training size", X_train.shape[0]
    print "Validation size", X_val.shape[0]
    print "Test size", X_test.shape[0]

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
    network = None
    if embeddings_file:
        network = build_model(vmap, n_classes, input_var=X, mask_var=M, ini_word2vec=True, word2vec_file=embeddings_file)
    else:
        network = build_model(vmap, n_classes, input_var=X, mask_var=M)

    if params_file is not None:
        print "Initializing params from file: %s" % params_file
        read_model_data(network, params_file)

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


def test_model(args):

    def load_vocab(vocab_file):
        vmap = {}
        with open(vocab_file, "r") as vf:
            for line in vf:
                # if len(line.split("\t")) > 3:
                #     id, _, _, cnt = line.strip().decode('utf-8').split("\t")
                #     w = r"\t"
                # else:
                id, w, cnt = line.strip().decode('utf-8').split("\t")
                vmap[w] = int(id)
        return vmap

    def get_files(semeval=False):
        datasets = {'train': None, 'dev': None, 'test': None}
        if semeval:
            root = '/home/kate/F15/semeval16/data/subtask-A'
            for i in ['train', 'dev', 'test']:
                datasets[i] = '%s/%s.tsv' % (root, i)
        else:
            root = '/home/kate/F15/semeval16/WORD_DATA/big/old'
            datasets['train'] = '%s/tweets.tsv.small' % root
            datasets['test'] = '%s/tweets.tsv.test' % root
        return datasets

    logfile = open('%s/results.txt' % args.logdir, 'w+')
    vmap = load_vocab(args.vocab)
    V = len(vmap)
    n_classes = 2
    logfile.write("vocab size: %d\n" % V)

    # Initialize theano variables for input and output
    X = T.imatrix('X')
    M = T.matrix('M')
    y = T.ivector('y')

    # Construct network
    print "Building Model"
    network = None
    if args.embeddings_file:
        logfile.write("embeddings file: %s" % args.embeddings_file)
        network = build_model(vmap, n_classes, input_var=X, mask_var=M, ini_word2vec=True, word2vec_file=args.embeddings_file)
    else:
        network = build_model(vmap, n_classes, input_var=X, mask_var=M)
    # Get network output
    output = lasagne.layers.get_output(network)

    # Define objective function (cost) to minimize, mean crossentropy error
    cost = lasagne.objectives.categorical_crossentropy(output, y).mean()

    # Compute gradient updates
    params = lasagne.layers.get_all_params(network)

    # need to switch off droput while testing
    test_output = lasagne.layers.get_output(network, deterministic=True)
    val_cost_fn = lasagne.objectives.categorical_crossentropy(test_output, y).mean()
    preds = T.argmax(test_output, axis=1)
    val_acc_fn = T.mean(T.eq(preds, y), dtype=theano.config.floatX)
    val_fn = theano.function([X, M, y], [val_cost_fn, val_acc_fn, preds], allow_input_downcast=True)

    def compute_val_error(log_file, X_val, y_val):
        batchsize = min(64, len(y_val))
        val_loss = 0.
        val_acc = 0.
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
            x_val_mini, y_val_mini = batch
            v_loss, v_acc, _ = val_fn(x_val_mini[:, :, 0], x_val_mini[:, :, 1], y_val_mini)
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

    logfile.write('load params from file: %s\n\n' % args.model_file)
    read_model_data(network, args.model_file)
    logfile.write('~~~ 1.6M ~~~\n')
    dsets = get_files(semeval=False)
    print "files:", dsets
    trainx, trainy, valx, valy, testx, testy, _ = load_big_dataset(dsets['train'], dsets['test'], args.vocab)
    logfile.write('train: %s %d\n' % (dsets['train'], testy.shape[0]))
    logfile.write('dev: None %d\n' % valy.shape[0])
    logfile.write('test: %s %d\n\n' % (dsets['test'], testy.shape[0]))
    logfile.flush()

    val_loss, val_acc = compute_val_error(logfile, valx, valy)
    test_loss, test_acc, _ = val_fn(testx[:, :, 0], testx[:, :, 1], testy)
    logfile.write("Best Model Test accuracy:\t\t{:.2f}%\n".format(test_acc * 100.))
    logfile.flush()

    dsets = get_files(semeval=True)
    trainx, trainy, valx, valy, testx, testy, _ = load_semeval_dataset(dsets['train'], dsets['test'], dsets['dev'], args.vocab)
    logfile.write('\n\n~~~ SEMEVAL ~~~\n')
    logfile.write('train: %s %d\n' % (dsets['train'], testy.shape[0]))
    logfile.write('dev: %s %d\n' % (dsets['dev'], valy.shape[0]))
    logfile.write('test: %s %d\n\n' % (dsets['test'], testy.shape[0]))
    logfile.flush()
    val_loss, val_acc = compute_val_error(logfile, valx, valy)
    test_loss, test_acc, _ = val_fn(testx[:, :, 0], testx[:, :, 1], testy)
    logfile.write("Best Model Test accuracy:\t\t{:.2f}%\n".format(test_acc * 100.))
    logfile.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='train word-level lstm')
    p.add_argument('--tweet-file', required=True, help='path to train data')
    p.add_argument('--vocab', required=True, help='path to vocabulary')
    p.add_argument('--log-path', type=str, help='path to store log file')
    p.add_argument('--logdir', type=str)

    p.add_argument('--test-file', help='path to test file')
    p.add_argument('--dev-file', type=str)

    p.add_argument('--embeddings-file', help='path to embeddings')

    p.add_argument('--nepochs', type=int, default=30, help='# of epochs')
    p.add_argument('--batchsize', type=int, default=512, help='batch size')
    p.add_argument('--learning-rate', type=float, default=0.1, help='learning rate')

    p.add_argument('--model-file', type=str)

    args = p.parse_args()
    print("ARGS:")
    print(args)

    test_model(args)

    # learn_model(
    #     train_path=args.tweet_file,
    #     vocab_file=args.vocab,
    #     test_path=args.test_file,
    #     num_epochs=args.nepochs,
    #     batchsize=args.batchsize,
    #     learn_rate=args.learning_rate,
    #     log_path=args.log_path,
    #     embeddings_file=args.embeddings_file
    # )


    # train_file = '../data/subtask-A/train.tsv'
    # val_file = '../data/subtask-A/dev.tsv'
    # test_file = '../data/subtask-A/test.tsv'
    # learn_model(train_file, val_file, test_file)

    # tweet_file = '/iesl/canvas/tbansal/trainingandtestdata/char_tweets_1.6M_processed_new_bow.tsv'
    # vfile = '/iesl/canvas/tbansal/trainingandtestdata/char_tweets_1.6M_processed_new_bow.tsv.vocab.txt'
    # test_file = '/iesl/canvas/tbansal/trainingandtestdata/char_tweets_1.6M_processed_new_bow.tsv.test.tsv'
    # learn_model(train_path=tweet_file, vocab_file=vfile, test_path=test_file,
    #             num_epochs=30, batchsize=512, learn_rate=0.1,
    #             log_path='lstm_result/char_bidirectional/finalDropout_256hu/')
