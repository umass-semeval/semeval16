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
    train_y = map(lambda tweet: classnames.index(tweet.label), train)
    dev_y = map(lambda tweet: classnames.index(tweet.label), dev)
    test_y = map(lambda tweet: classnames.index(tweet.label), test)

    fx = FeatureExtractor(["word2vec"], word2vec_model=args.word2vec_model)
    fx.build_vocab(train)
    train_x = np.asarray(map(lambda tweet: fx.process(tweet), train))
    check = train_x[0]
    print("sample fv shape: ", check.shape)
    dev_x = np.asarray(map(lambda tweet: fx.process(tweet), dev))
    test_x = np.asarray(map(lambda tweet: fx.process(tweet), test))

    nclasses = len(classnames)
    train_data = (train_x, train_y)
    dev_data = (dev_x, dev_y)
    test_data = (test_x, test_y)

    classifier = neural_net.train_mlp(train_data, dev_data, test_data, nclasses,
                                      nlayers=2, units=[512, 256],
                                      learning_rate=0.001, L2_reg=0.1,
                                      batch_size=60)

    print("train set performance:")
    train_ypred = neural_net.predict2(classifier, train_x, train_y)
    print(evaluate.ConfusionMatrix(train_y, train_ypred, classnames))

    print("validation set performance:")
    dev_ypred = neural_net.predict2(classifier, dev_x, dev_y)
    print(evaluate.ConfusionMatrix(dev_y, dev_ypred, classnames))

    print("test set performance:")
    test_ypred = neural_net.predict2(classifier, test_x, test_y)
    print(evaluate.ConfusionMatrix(test_y, test_ypred, classnames))


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


