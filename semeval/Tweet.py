class Tweet:
    def __init__(self, id, label, raw_text, topic=None):
        self.id = id
        self.label = label
        self.raw_text = raw_text
        self.label_intval = -1
        self.topic = topic

    def __repr__(self):
        return "Tweet(%s\t%s\t%s\t%s)" % (self.label, self.raw_text, self.id, self.topic)


def parse_line(line, subtask_id='a'):
    parts = line.split('\t')
    if subtask_id == 'a':
        if len(parts) == 3:
            return Tweet(parts[0], parts[1], parts[2].strip("\n"))
        else:
            print("bad line: %s" % line)
            return None
    elif subtask_id == 'b':
        if len(parts) == 4:
            tid = parts[0]
            topic = parts[1]
            label = parts[2]
            text = parts[3].strip("\n")
            return Tweet(tid, label, text, topic=topic)
        else:
            print("bad line: %s" % line)
            return None


def load_from_tsv(filename, subtask_id='a'):
    tweets = []
    with open(filename, "r") as f:
        for line in f.readlines():
            tweet = parse_line(line, subtask_id=subtask_id)
            if tweet is not None:
                tweets.append(tweet)
    tweets_clean = []
    errors = 0
    """
    Filter out tweets that failed to download properly
    """
    for tweet in tweets:
        if tweet.raw_text == "Not Available":
            errors += 1
        else:
            tweets_clean.append(tweet)
    print("loaded %d / %d tweets from file %s; %d failed to download properly." %
          (len(tweets_clean), len(tweets), filename, errors))
    return tweets_clean


def load_datasets(args):
    subtask_id = args.subtask_id
    train = load_from_tsv(args.train_file, subtask_id=subtask_id)
    dev = load_from_tsv(args.dev_file, subtask_id=subtask_id)
    if args.test_file is not None:
        test = load_from_tsv(args.test_file, subtask_id=subtask_id)
    else:
        ndev = len(dev)
        test_portion = int(0.5 * ndev)
        test = dev[:test_portion]
        dev = dev[test_portion:]
    print("ntrain: %d, ndev: %d, ntest: %d" % (len(train), len(dev), len(test)))
    return train, dev, test