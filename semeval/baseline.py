from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from Tweet import *
from FeatureExtractor import *
import argparse
import neural_net
import evaluate


def classmap(tweets):
    classes = get_classes(tweets)
    class_map = {}
    i = 0
    for c in classes:
        class_map[c] = i
        i += 1
    return classes, class_map


def get_classes(tweets):
    return list(set(map(lambda x: x.label, tweets)))


def extract_features(tweets, count_vec, tt, is_test):
    strings = map(lambda x: x.raw_text, tweets)
    x_tfidf = None
    if not is_test:
        x_counts = count_vec.fit_transform(strings)
        x_tfidf = tt.fit_transform(x_counts)
    else:
        x_counts = count_vec.transform(strings)
        x_tfidf = tt.transform(x_counts)
    return x_tfidf


def accuracy(tweets, prediction, classes, extra="", verbose=False):
    ncorrect = 0.0
    total = 0.0
    for tweet, category in zip(tweets, prediction):
        predicted = classes[category]
        true_class = tweet.label
        correct = ""
        if predicted == true_class:
            correct = ""
            ncorrect += 1.0
        else:
            correct = "*"
        total += 1.0
        if verbose:
            print("%s\t%s %s : %s" % (correct, true_class, predicted, tweet.raw_text))
    acc = ncorrect/total
    print("%s accuracy: %f (%d/%d)" % (extra, acc, int(ncorrect), int(total)))
    return acc


def run_naivebayes_baseline(args):
    print(args)
    train_tweets = load_from_tsv(args.train_file)
    dev_tweets = load_from_tsv(args.dev_file)
    classes, cmap = classmap(train_tweets)
    count_vec = CountVectorizer()
    tt = TfidfTransformer()
    x_train = extract_features(train_tweets, count_vec, tt, False)
    y_train = map(lambda x: cmap[x.label], train_tweets)
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_train)
    accuracy(train_tweets, prediction, classes, extra="train")
    x_test = extract_features(dev_tweets, count_vec, tt, True)
    test_prediction = clf.predict(x_test)
    accuracy(dev_tweets, test_prediction, classes, extra="test")


def run_BOW_baseline(args):
    print(args)
    train = load_from_tsv(args.train_file)
    dev = load_from_tsv(args.dev_file)
    test = load_from_tsv(args.test_file)
    print("ntrain: %d, ndev: %d, ntest: %d" % (len(train), len(dev), len(test)))
    classnames = list(set(map(lambda tweet: tweet.label, train)))
    train_y = map(lambda tweet: classnames.index(tweet.label), train)
    dev_y = map(lambda tweet: classnames.index(tweet.label), dev)
    test_y = map(lambda tweet: classnames.index(tweet.label), test)
    fx = FeatureExtractor(["BOW"], stopwords=args.stopwords)
    fx.build_vocab(train)
    train_x = np.asarray(map(lambda tweet: fx.process(tweet), train))
    check = train_x[0]
    print("sample fv shape: ", check.shape)
    dev_x = np.asarray(map(lambda tweet: fx.process(tweet), dev))
    test_x = np.asarray(map(lambda tweet: fx.process(tweet), test))
    nclasses = len(classnames)
    ntrain = train_x.shape[0]
    nbatches = 100
    batch_size = ntrain/nbatches
    train_data = (train_x, train_y)
    dev_data = (dev_x, dev_y)
    test_data = (test_x, test_y)
    neural_net.logistic_regression_optimization_sgd(train_data, dev_data, test_data, nclasses, batch_size=batch_size)
    print("train set performance:")
    train_ypred = neural_net.predict(train_x, train_y)
    print(evaluate.ConfusionMatrix(train_y, train_ypred, classnames))
    print("validation set performance:")
    dev_ypred = neural_net.predict(dev_x, dev_y)
    print(evaluate.ConfusionMatrix(dev_y, dev_ypred, classnames))
    print("test set performance:")
    test_ypred = neural_net.predict(test_x, test_y)
    print(evaluate.ConfusionMatrix(test_y, test_ypred, classnames))


def run_word2vec_baseline(args):
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
    ntrain = train_x.shape[0]
    nbatches = 100
    batch_size = ntrain/nbatches
    train_data = (train_x, train_y)
    dev_data = (dev_x, dev_y)
    test_data = (test_x, test_y)
    neural_net.logistic_regression_optimization_sgd(train_data, dev_data, test_data, nclasses, batch_size=batch_size)
    print("train set performance:")
    train_ypred = neural_net.predict(train_x, train_y)
    print(evaluate.ConfusionMatrix(train_y, train_ypred, classnames))
    print("validation set performance:")
    dev_ypred = neural_net.predict(dev_x, dev_y)
    print(evaluate.ConfusionMatrix(dev_y, dev_ypred, classnames))
    print("test set performance:")
    test_ypred = neural_net.predict(test_x, test_y)
    print(evaluate.ConfusionMatrix(test_y, test_ypred, classnames))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument('--model-type', help='naive_bayes|bow_logistic_regression')
    parser.add_argument('--train-file', help='file for train data')
    parser.add_argument('--dev-file', help='file for dev data')
    parser.add_argument('--test-file', help='file for dev data')
    parser.add_argument('--stopwords', help='file for stopwords')
    parser.add_argument('--word2vec-model', help='path to word2vec vectors')
    args = parser.parse_args()
    if args.model_type == "naive_bayes":
        run_naivebayes_baseline(args)
    elif args.model_type == "bow_logistic_regression":
        run_BOW_baseline(args)
    elif args.model_type == "word2vec_logistic_regression":
        run_word2vec_baseline(args)
    else:
        raise Exception("invalid model-type: %s" % args.model_type)




