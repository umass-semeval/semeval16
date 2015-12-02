from gensim import corpora
import csv
import re
import argparse


class MyCorpus(object):
    def __init__(self, fname, stopf=None, V=None, dictionary=None):
        self.fname = fname
        self.file = open(fname, "r")
        stoplist = []
        if stopf:  # read stop words
            with open(stopf, 'r') as f:
                stoplist = map(lambda x: x.strip().lower(), f.readlines())
        if not dictionary:
            self.dictionary = self.make_dict(stoplist, V)
        else:
            self.dictionary = dictionary

    def reset(self):
        self.file.seek(0)

    def proc(self, line):
        return filter(lambda x: len(x) >= 2,
                      map(lambda x: x.strip(),
                      re.sub(r'[0-9]+|\W', ' ',
                             line.strip().lower()).split()))

    def make_dict(self, stoplist=[], V=None, no_below=30, no_above=0.5):
        self.reset()
        # read all terms
        dictionary = corpora.Dictionary(self.proc(line)
                                        for _, line, _ in self.read_file()
                                        )

        ''' remove words which occur in less than 5
            documents or more than 50% of documents
        '''
        dictionary.filter_extremes(no_below=no_below, no_above=no_above,
                                   keep_n=V)

        # remove stop words
        stop_ids = [dictionary.token2id[sw]
                    for sw in stoplist if sw in dictionary.token2id
                    ]
        dictionary.filter_tokens(stop_ids)

        # remove gaps
        dictionary.compactify()

        return dictionary

    def read_file(self):
        # self.reset()
        csvreader = csv.reader(self.file, delimiter=',')
        for line in csvreader:
            tweet_id = line[1].strip()
            txt = line[5].strip()
            label = int(line[0].strip())
            yield tweet_id, txt, label

    def __iter__(self):
        self.reset()
        for tid, line, label in self.read_file():
            # bow = self.dictionary.doc2bow(self.proc(line))
            tokenized_line = []
            for w in self.proc(line):
                if w in self.dictionary.token2id:
                    tokenized_line.append(self.dictionary.token2id[w])
            yield tid, tokenized_line, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process 1.6M tweet corpus")
    parser.add_argument('--tweet-file', help="location of 1.6M corpus",
                        required=True)
    parser.add_argument('--output-file', help="location, name of output",
                        required=True)
    parser.add_argument('--test-file', help="location of test file",
                        default=None)
    parser.add_argument('--stop-words', help="location of stopwords",
                        default=None)
    parser.add_argument('--V', help="vocab size", default=None)

    args = parser.parse_args()
    tweet_corpus = MyCorpus(args.tweet_file, stopf=args.stop_words, V=args.V)

    print tweet_corpus.dictionary

    tweet_corpus.dictionary.save_as_text(args.output_file+".vocab.txt",
                                         sort_by_word=False)

    def read_proc(outname, corpus):
        num_blank = 0
        label_map = {0: "negative", 2: "neutral", 4: "positive"}
        with open(outname, "w+") as outf:
            for tid, tweet_bow, senti in corpus:
                if len(tweet_bow) == 0 or senti == 4:
                    # NOTE: will ignore neutral tweets
                    num_blank += 1
                    continue
                outf.write(tid + "\t" + label_map[senti] + "\t" +
                           " ".join(map(str, tweet_bow)) + "\n")
        return num_blank

    num_blank = read_proc(args.output_file, tweet_corpus)
    print "Removed Tweets in train/val set", num_blank

    if args.test_file:
        print "Processing test set"
        test_corpus = MyCorpus(args.test_file,
                               dictionary=tweet_corpus.dictionary)
        num_blank = read_proc(args.output_file + ".test.tsv", test_corpus)
        print "Removed Tweets in test set", num_blank

