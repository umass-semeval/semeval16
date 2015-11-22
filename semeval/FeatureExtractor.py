import utils
import numpy as np
import embedding

class FeatureExtractor:

    def __init__(self, template_ids, **kwargs):
        self.template_ids = template_ids
        self.vocab = None
        self.stopwords = set([])
        if "stopwords" in kwargs:
            self.stopwords = utils.load_stopwords(kwargs["stopwords"])
            print("loaded %d stopwords" % len(self.stopwords))
        self.word2vec_model = None
        if "word2vec_model" in kwargs:
            self.word2vec_model = embedding.load_embeddings(kwargs["word2vec_model"])
            print("loaded word2vec model from %s" % kwargs["word2vec_model"])
        self.embedding_dim = None

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
        fv = None
        for tid in self.template_ids:
            if tid == "BOW":
                fv = self.process_BOW(tweet)
            if tid == "word2vec":
                fv = self.process_word2vec(tweet)
            if tid == "hand-coded":
                fv = self.process_hand_coded_features(tweet)
        return fv

    def process_hand_coded_features(self, tweet):
        words = self._tokenize(tweet)
        ncaps = len(filter(lambda w: w.istitle(), words))
        nwords = self._normalize(words)
        fv = [0]*200
        fv[0] = ncaps
        n = len(nwords)
        for i in range(1, n):
            w = nwords[i]
            if w in self.vocab:
                fv[i] = self.vocab.index(w)
        return fv

    def process_BOW(self, tweet):
        V = len(self.vocab)
        words = self._tokenize(tweet)
        nwords = self._normalize(words)
        n = len(nwords)
        BOW = [0]*V
        for i in range(n):
            try:
                idx = self.vocab.index(nwords[i])
                BOW[idx] = 1
            except ValueError:
                pass
        return np.asarray(BOW)

    def process_word2vec(self, tweet):
        nwords = self._normalize(self._tokenize(tweet))
        if self.embedding_dim is None:
            for w in nwords:
                if w in self.word2vec_model.vocab:
                    vec = self.word2vec_model[w]
                    self.embedding_dim = len(vec)
                    print("set embedding dim to %d" % self.embedding_dim)
                    break
        x = np.zeros(self.embedding_dim)
        for w in nwords:
            if w in self.word2vec_model.vocab:
                x += self.word2vec_model[w]
        return x

    def process_word2vec_noagg(self, tweet, window_size):
        nwords = self._normalize(self._tokenize(tweet))
        if self.embedding_dim is None:
            for w in nwords:
                if w in self.word2vec_model.vocab:
                    vec = self.word2vec_model[w]
                    self.embedding_dim = len(vec)
                    print("set embedding dim to %d" % self.embedding_dim)
                    break
        x = []
        for w in nwords:
            if w in self.word2vec_model.vocab:
                x.append(self.word2vec_model[w])
        n = len(x)
        if n < window_size:
            for i in range(window_size - n):
                x.append(np.zeros(self.embedding_dim))
            matrix = np.asarray(x)
            assert matrix.shape[0] == window_size, "shape[0] %d != window size %d" % (matrix.shape[0], window_size)
            return matrix, None
        elif n > window_size:
            x1 = []
            i = 0
            while i < window_size:
                x1.append(x[i])
                i += 1
            left = window_size - len(x1)
            x2 = []
            while i < left:
                x2.append(x[i])
                i += 1
            if len(x2) < window_size:
                for i in range(window_size - len(x2)):
                    x2.append(np.zeros(self.embedding_dim))
            m1, m2 = np.asarray(x1), np.asarray(x2)
            assert m1.shape[0] == window_size
            assert m2.shape[0] == window_size
            return m1, m2
        else:
            return np.asarray(x), None


    @staticmethod
    def _tokenize(tweet):
        return tweet.raw_text.split(" ")

    @staticmethod
    def _normalize(words):
        lowercase = map(lambda w: w.lower(), words)
        clean = filter(lambda w: len(w) > 0, lowercase)
        return clean