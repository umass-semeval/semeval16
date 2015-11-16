import numpy as np

class Preprocessor:
	def __init__(self):
		self.vocab = None
		self.padding_constant = "<PAD>"
		self.padding_idx = -1
	def build_vocab(self, tweets):
		vocab = set([])
		for t in tweets:
			words = self._tokenize(t)
			nwords = self._normalize(words)
			for w in nwords:
				vocab.add(w)
		self.vocab = list(vocab)
		self.vocab.append(self.padding_constant)
		self.padding_idx = self.vocab.index(self.padding_constant)
		print("initialized vocab with %d words" % len(vocab))
	
	def preprocess(self, tweet):
		words = self._tokenize(tweet)
		nwords = self._normalize(words)
		ints = map(lambda w: self.vocab.index(w), nwords)
		return np.asarray(ints)

	def preprocess_constant_len(self, tweet, n):
		ints = self.preprocess(tweet)
		diff = n - ints.shape[0]
		if diff > 0:
			padding_amt = int(float(diff)/2.0)
			ints = np.pad(ints, padding_amt, mode='constant', constant_values=self.padding_idx)
		if ints.shape[0] == n - 1:
			ints = np.append(ints, self.padding_idx)					
		assert ints.shape[0] == n, "%d" % ints.shape[0]
		return ints

	def _tokenize(self, tweet):
		return tweet.raw_text.split(" ")
	def _normalize(self, words):
		return map(lambda w: w.lower(), words)
		
