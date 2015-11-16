import utils
import numpy as np


class FeatureExtractor:

    def __init__(self, template_ids, **kwargs):
        self.template_ids = template_ids
        self.vocab = None
        self.stopwords = set([])
        if "stopwords" in kwargs:
            self.stopwords = utils.load_stopwords(kwargs["stopwords"])
            print("loaded %d stopwords" % len(self.stopwords))

    def build_vocab(self, tweets):
        counts = {}
        for t in tweets:
            words = filter(lambda w: not w in self.stopwords, self._normalize(self._tokenize(t)))
            for w in words:
                if not w in counts:
                    counts[w] = 0
                counts[w] += 1
        self.vocab = counts.keys()

    def process(self, tweet):
        V = len(self.vocab)
        words = self._tokenize(tweet)
        contains_cap = 0
        for word in words:
            if word.title():
                contains_cap = 1
                break
        nwords = self._normalize(words)
        n = len(nwords)
        ints = [0]*V
        for i in range(n):
            try:
                idx = self.vocab.index(nwords[i])
                ints[idx] = 1
            except ValueError:
                pass
        fv = [contains_cap] + ints
        return np.array(fv)

    @staticmethod
    def _tokenize(tweet):
        return tweet.raw_text.split(" ")

    @staticmethod
    def _normalize(words):
        lowercase = map(lambda w: w.lower(), words)
        clean = filter(lambda w: len(w) > 0, lowercase)
        return clean