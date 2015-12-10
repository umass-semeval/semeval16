import twokenize
import re


URL_PATTERN = re.compile(ur'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
NUM_PATTERN = re.compile(ur'\d+')

def tokenize_tweet(text):
    return twokenize.tokenize(text)


def normalize_tweet(text, lowercase=False, rm_digits=False, return_tokens=False):
    if lowercase:
        text = text.lower()
    text = re.sub(URL_PATTERN, 'URL', text)
    tokens = twokenize.tokenize(text)
    if return_tokens:
        if rm_digits:
            tokens = map(lambda tk: re.sub(NUM_PATTERN, 'NUM', tokens))
        return tokens
    clean = ' '.join(tokens)
    if rm_digits:
        re.sub(NUM_PATTERN, 'NUM', clean)
    return clean


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


