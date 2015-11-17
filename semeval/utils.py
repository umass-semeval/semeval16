def split_data(tweets, train_portion=0.8):
    n = float(len(tweets))
    trainp = int(train_portion * n)
    train = tweets[:trainp]
    test = tweets[trainp:]
    return (train,test)


def write_stopwords():
    from nltk.corpus import stopwords
    stops = stopwords.words('english')
    outfile = "stopwords.txt"
    with open(outfile, "w") as f:
        for w in stops:
            f.write(w + "\n")


def load_stopwords(path):
    stopwords = set([])
    with open(path, "r") as f:
        for line in f.readlines():
            stopwords.add(line.rstrip())
    return stopwords
