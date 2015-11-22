import argparse
import numpy as np
from Tweet import *
from FeatureExtractor import *
import neural_net
import evaluate


def corpus_stats(tweets):
    classes = set(map(lambda x: x.label, tweets))
    print("classes: %s" % ",".join(list(classes)))
    for c in classes:
        examples = filter(lambda x: x.label == c, tweets)
        print("class %s: %d tweets total" % (c, len(examples)))


def set_intvals(tweets, classnames):
    for tw in tweets:
        tw.label_intval = classnames.index(tw.label)


def main(args):
    print(args)
    train = load_from_tsv(args.train_file)
    dev = load_from_tsv(args.dev_file)
    test = load_from_tsv(args.test_file)
    print("ntrain: %d, ndev: %d, ntest: %d" % (len(train), len(dev), len(test)))

    classnames = list(set(map(lambda tweet: tweet.label, train)))
    
    window_size = 20

    print("computing features (train)")
    fx = FeatureExtractor(["word2vec"], word2vec_model=args.word2vec_model)
    train_y = []
    train_x = []
    for tw in train:
        label = classnames.index(tw.label)
        m1,m2 = fx.process_word2vec_noagg(tw, window_size)
        train_x.append(m1)
        train_y.append(label)
        if m2 is not None:
            train_x.append(m2)
            train_y.append(label)
    train_y = np.asarray(train_y)
    train_x = np.asarray(train_x)
    print("train_x: ", train_x.shape)

    print("computing features (dev)")
    dev_y = []
    dev_x = []
    for tw in dev:
        label = classnames.index(tw.label)
        m1,m2 = fx.process_word2vec_noagg(tw, window_size)
        dev_x.append(m1)
        dev_y.append(label)
        if m2 is not None:
            dev_x.append(m2)
            dev_y.append(label)
    dev_y = np.asarray(dev_y)
    dev_x = np.asarray(dev_x)

    print("computing features (test)")
    test_y = []
    test_x = []
    for tw in test:
        label = classnames.index(tw.label)
        m1,m2 = fx.process_word2vec_noagg(tw, window_size)
        test_x.append(m1)
        test_y.append(label)
        if m2 is not None:
            test_x.append(m2)
            test_y.append(label)
    test_y = np.asarray(test_y)
    test_x = np.asarray(test_x)
    print("done")

    nclasses = len(classnames)
    train_data = (train_x, train_y)
    dev_data = (dev_x, dev_y)
    test_data = (test_x, test_y)

    neural_net.train_cnn(train_data, dev_data, test_data, nclasses, window_size=window_size)

    # classifier = neural_net.train_mlp(train_data, dev_data, test_data, nclasses,
    #                                   nlayers=2, units=[512, 256],
    #                                   learning_rate=0.001, L2_reg=0.1,
    #                                   batch_size=60)
    #
    # print("train set performance:")
    # train_ypred = neural_net.predict2(classifier, train_x, train_y)
    # print(evaluate.ConfusionMatrix(train_y, train_ypred, classnames))
    #
    # print("validation set performance:")
    # dev_ypred = neural_net.predict2(classifier, dev_x, dev_y)
    # print(evaluate.ConfusionMatrix(dev_y, dev_ypred, classnames))
    #
    # print("test set performance:")
    # test_ypred = neural_net.predict2(classifier, test_x, test_y)
    # print(evaluate.ConfusionMatrix(test_y, test_ypred, classnames))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument('--model-type', help='naive_bayes|bow_logistic_regression')
    parser.add_argument('--train-file', help='file for train data')
    parser.add_argument('--dev-file', help='file for dev data')
    parser.add_argument('--test-file', help='file for dev data')
    parser.add_argument('--stopwords', help='file for stopwords')
    parser.add_argument('--word2vec-model', help='path to pre-trained word2vec vectors')
    args = parser.parse_args()
    main(args)

"""
def main(args):
    print(args)
    train = load_from_tsv(args.train_file)
    dev = load_from_tsv(args.dev_file)
    test = load_from_tsv(args.test_file)
    print("ntrain: %d, ndev: %d, ntest: %d" % (len(train), len(dev), len(test)))

    classnames = list(set(map(lambda tweet: tweet.label, train)))
    # train_y = map(lambda tweet: classnames.index(tweet.label), train)
    dev_y = map(lambda tweet: classnames.index(tweet.label), dev)
    test_y = map(lambda tweet: classnames.index(tweet.label), test)

    window_size = 20

    fx = FeatureExtractor(["word2vec"], word2vec_model=args.word2vec_model)
    fx.build_vocab(train)
    # train_x = np.asarray(map(lambda tweet: fx.process(tweet), train))
    # train_x = np.asarray(map(lambda tweet: fx.process_word2vec_noagg(tweet, window_size), train))
    train_y = []
    train_x = []
    for tw in train:
        label = classnames.index(tw.label)
        m1,m2 = fx.process_word2vec_noagg(tw, window_size)
        train_x.append(m1)
        train_y.append(label)
        if m2 is not None:
            train_x.append(m2)
            train_y.append(label)
    for i in range(5):
        print("sample fv shape:", train_x[i].shape)
    train_y = np.asarray(train_y)
    train_x = np.asarray(train_x)
    # # check = train_x[0]
    # # print("sample fv shape: ", check.shape)
    # # dev_x = np.asarray(map(lambda tweet: fx.process(tweet), dev))
    # # test_x = np.asarray(map(lambda tweet: fx.process(tweet), test))
    # dev_x = np.asarray(map(lambda tweet: fx.process_word2vec_noagg(tweet, window_size), dev))
    # test_x = np.asarray(map(lambda tweet: fx.process_word2vec_noagg(tweet, window_size), test))

    print("train_x: ", train_x.shape)

    # nclasses = len(classnames)
    # train_data = (train_x, train_y)
    # dev_data = (dev_x, dev_y)
    # test_data = (test_x, test_y)
    #
    # neural_net.train_cnn(train_data, dev_data, test_data, nclasses, window_size=window_size)

    # classifier = neural_net.train_mlp(train_data, dev_data, test_data, nclasses,
    #                                   nlayers=2, units=[512, 256],
    #                                   learning_rate=0.001, L2_reg=0.1,
    #                                   batch_size=60)
    #
    # print("train set performance:")
    # train_ypred = neural_net.predict2(classifier, train_x, train_y)
    # print(evaluate.ConfusionMatrix(train_y, train_ypred, classnames))
    #
    # print("validation set performance:")
    # dev_ypred = neural_net.predict2(classifier, dev_x, dev_y)
    # print(evaluate.ConfusionMatrix(dev_y, dev_ypred, classnames))
    #
    # print("test set performance:")
    # test_ypred = neural_net.predict2(classifier, test_x, test_y)
    # print(evaluate.ConfusionMatrix(test_y, test_ypred, classnames))
"""