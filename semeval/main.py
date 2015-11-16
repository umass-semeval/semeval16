import sys
import argparse
from Tweet import *
from utils import *
from FeatureExtractor import *
import neural_net

import numpy as np


def corpus_stats(tweets):
    classes = set(map(lambda x: x.label, tweets))
    print("classes: %s" % ",".join(list(classes)))
    for c in classes:
        examples = filter(lambda x: x.label == c, tweets)
        print("class %s: %d tweets total" % (c, len(examples)))


def main(args):
    print(args)
    filename = args.train_file
    tweets = load_from_tsv(filename)
    for t in tweets[:10]:
        print(t)
    corpus_stats(tweets)
    if args.dev_file is not None:
        train = tweets
        dev = load_from_tsv(args.dev_file)
    else:
        train, dev = split_data(tweets)
    print("ntrain: %d, ndev: %d" % (len(train), len(dev)))
    
    classnames = list(set(map(lambda tweet: tweet.label, train)))

    stopwords_path = None
    if args.stopwords is not None:
        stopwords_path = args.stopwords
    fx = FeatureExtractor([], stopwords=stopwords_path)
    fx.build_vocab(train)

    train_x = np.asarray(map(lambda tweet: fx.process(tweet), train))
    train_y = np.asarray(map(lambda tweet: classnames.index(tweet.label), train))

    dev_y = np.asarray(map(lambda tweet: classnames.index(tweet.label), dev))
    dev_x = np.asarray(map(lambda tweet: fx.process(tweet), dev))

    print("train", train_x.shape, train_y.shape)
    print("dev", dev_x.shape, dev_y.shape)

    ntrain = train_y.shape[0]
    nbatches = 100
    batch_size = ntrain/nbatches
    print("batch size: %d" % batch_size)

    ncorrect, total = neural_net.train_mlp(train_x, train_y, dev_x, dev_y, len(classnames), batch_size=batch_size)
    # ncorrect, total = neural_net.predict(dev_x, dev_y)
    acc = ncorrect / total
    print(ncorrect, total, acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument('--train-file', help='file for train data')
    parser.add_argument('--dev-file', help='file for dev data')
    parser.add_argument('--stopwords', help='file for stopwords')
    args = parser.parse_args()
    # args = sys.argv[1:]
    main(args)


