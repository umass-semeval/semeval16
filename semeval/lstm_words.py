import numpy as np
import theano
import theano.tensor as T

import time
import Tweet

import lasagne
from gensim.models import word2vec

import os
import pickle

MAX_SEQ = 70  # maximum length of a sequence


def build_model(vmap,  # input vocab mapping
                num_classes,  # number classes to predict
                K=20,  # dimensionality of embeddings
                num_hidden=32,  # number of hidden_units
                batchsize=None,  # size of each batch (None for variable size)
                input_var=None,  # theano variable for input
                mask_var=None,  # theano variable for input mask
                grad_clip=5,  # gradients above this will be clipped
                max_seq_len=MAX_SEQ,  # maximum lenght of a sequence
                ini_word2vec=True,  # whether to initialize with word2vec
                word2vec_file='/iesl/canvas/tbansal/glove.twitter.27B.50d.txt',
                # location of trained word vectors
                ):

    V = len(vmap)
    # basic input layer (batchsize, SEQ_LENGTH),
    # None lets us use variable bs
    # use a mask to outline the actual input sequence
    if ini_word2vec:
        word2vec_model = word2vec.Word2Vec.load_word2vec_format(word2vec_file,
                                                                binary=False)
        print "  Loaded Word2Vec model"
        K = word2vec_model[word2vec_model.vocab.keys()[0]].size  # override dim
        W = np.zeros((V, K), dtype=np.float32)
        no_vectors = 0
        for w in vmap:
            if w in word2vec_model.vocab:
                W[vmap[w]] = word2vec_model[w]
            else:
                W[vmap[w]] = np.random.normal(scale=0.01, size=K)
                no_vectors += 1
        print "  Initialized with word2vec. Couldn't find", no_vectors, "words!"
    else:
        W = lasagne.init.Normal()

    l_in = lasagne.layers.InputLayer((batchsize, max_seq_len),
                                     input_var=input_var)

    l_mask = lasagne.layers.InputLayer((batchsize, max_seq_len),
                                       input_var=mask_var)

    l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=V,
                                          output_size=K, W=W)

    # add droput
    l_emb = lasagne.layers.DropoutLayer(l_emb, p=0.4)

    print lasagne.layers.get_output_shape(l_emb,
                                          {l_in: (200, 140),
                                           l_mask: (200, 140)})

    # network = lasagne.layers.LSTMLayer(
    #     l_emb, num_units=num_hidden, grad_clipping=grad_clip,
    #     nonlinearity=lasagne.nonlinearities.tanh, mask_input=l_mask
    # )
    #
    # print lasagne.layers.get_output_shape(network,
    #                                       {l_in: (200, 140),
    #                                        l_mask: (200, 140)})
    #
    # # add droput
    # network = lasagne.layers.DropoutLayer(network, p=0.8)

    # second hidden layer
    # using just the activation at last time step as the representation
    network = lasagne.layers.LSTMLayer(
        l_emb, num_units=num_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh, mask_input=l_mask,
        only_return_final=True
    )

    print lasagne.layers.get_output_shape(network,
                                          {l_in: (200, 140),
                                           l_mask: (200, 140)})

    # add droput
    network = lasagne.layers.DropoutLayer(network, p=0.9)

    # output is dense layer (over all hidden units?)
    network = lasagne.layers.DenseLayer(
        network, num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax
    )

    print lasagne.layers.get_output_shape(network,
                                          {l_in: (200, 140),
                                           l_mask: (200, 140)})

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
    with open(filename, 'w') as f:
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
    ''' Use Kate's Tweet class '''

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


def load_big_dataset(tweet_file, vocab_file, val_ratio=0.05):
    '''
        Load the big 1.6 million Dataset
        tweet_file is location of the processed tweets
        val_ratio is the fraction of tweets required for validation
    '''
    vmap = {}
    with open(vocab_file, "r") as vf:
        for line in vf:
            id, w, cnt = line.strip().split("\t")
            vmap[w.strip()] = int(id)

    label_map = {'negative': 0,
                 'positive': 1}
    X = []
    y = []
    with open(tweet_file, "r") as tf:
        for line in tf:
            _, label, tweet = line.strip().split("\t")
            X.append(map(int, tweet.strip().split()))
            y.append(label_map[label.strip()])

    X = pad_mask(X)
    y = np.asarray(y, dtype=np.int32)
    n = X.shape[0]
    n_val = int(val_ratio * n)
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_indices = indices[:n-n_val]
    val_indices = indices[n-n_val:]

    return (X[train_indices], y[train_indices],
            X[val_indices], y[val_indices], vmap)


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


def learn_model(train_path, val_path=None, test_path=None, max_norm=5,
                num_epochs=5, batchsize=64, learn_rate=0.1,
                vocab_file=None, val_ratio=0.05):
    '''
        train to classify sentiment
        data is a tuple of (X, y)
            where X is (Num Ex, Max Seq Length, 2)
            with X[:,:,1] as the input mask
            and y is (Num Ex,)
        V is the vocab (or charset) size

        Returns the trained network
    '''

    print "Loading Dataset"
    # X_train, y_train, X_val, y_val, X_test, vmap = load_dataset(train_path,
    #                                                             val_path,
    #                                                             test_path)

    X_train, y_train, X_val, y_val, vmap = load_big_dataset(train_path,
                                                            vocab_file,
                                                            val_ratio)

    print "Training size", X_train.shape[0]
    print "Validation size", X_val.shape[0]
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
    network = build_model(vmap, n_classes, input_var=X, mask_var=M)

    # Get network output
    output = lasagne.layers.get_output(network)

    # Define objective function (cost) to minimize, mean crossentropy error
    cost = lasagne.objectives.categorical_crossentropy(output, y).mean()

    # Compute gradient updates
    params = lasagne.layers.get_all_params(network)
    # grad_updates = lasagne.updates.nesterov_momentum(cost, params,learn_rate)
    # grad_updates = lasagne.updates.adam(cost, params)
    grad_updates = lasagne.updates.adadelta(cost, params, learn_rate)

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

            if train_batches % 128 == 0:
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
                print("\tBatch {} of epoch {} took {:.3f}s".format(
                    train_batches, epoch+1, time.time() - start_time))
                print("\t training loss:\t\t{:.6f}".format(train_err /
                                                           train_batches))
                print("\t  validation loss:\t\t{:.6f}".format(val_loss /
                                                              val_batches))
                print("\t  validation accuracy:\t\t{:.2f} %".format(
                    val_acc / val_batches * 100.))
                if val_acc / val_batches >= best_val_acc:
                    best_val_acc = val_acc / val_batches
                    write_model_data(network, 'lstm_result/best_lstm_model')


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
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err /
                                                  train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_loss /
                                                    val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100.))
        if val_acc / val_batches >= best_val_acc:
            best_val_acc = val_acc / val_batches
            write_model_data(network, 'lstm_result/best_lstm_model')

        # print ("preds  0:", (val_err[2] == 0).sum(), "1:",
        #        (val_err[2] == 1).sum(), "2:", (val_err[2] == 2).sum()
        #        )
        # print ("truth  0:", (y_val == 0).sum(), "1:",
        #        (y_val == 1).sum(), "2:", (y_val == 2).sum()
        #        )
    print "Training took {:.3f}s".format(time.time() - begin_time)

    return network


if __name__ == "__main__":
    # train_file = '../data/subtask-A/train.tsv'
    # val_file = '../data/subtask-A/dev.tsv'
    # test_file = '../data/subtask-A/test.tsv'
    # learn_model(train_file, val_file, test_file)

    tweet_file = '/iesl/canvas/tbansal/trainingandtestdata/tweets_1.6M_processed_bow.tsv'
    vfile = '/iesl/canvas/tbansal/trainingandtestdata/tweets_1.6M_processed_bow.tsv.vocab.txt'
    learn_model(train_path=tweet_file, vocab_file=vfile, num_epochs=200,
                batchsize=1024, learn_rate=0.1)
