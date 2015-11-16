from sklearn.feature_extraction.text import CountVectorizer

class Corpus:
	def __init__(self, tweets):
		self.tweets = tweets
		self.vocab_size = -1
		self.cv = CountVectorizer()
		self.tokenizer = None
		self.build_vocab()
	def build_vocab(self):
		strings = map(lambda tweet: tweet.raw_text, self.tweets)
		self.cv.fit_transform(strings)
		self.tokenizer = self.cv.build_analyzer()		
		self.vocab_size = len(self.cv.vocabulary_.keys())
		print("vocabulary size: %d" % self.vocab_size)
	def vocab(self):
		return self.cv.vocabulary_
	def tweet2array(self, tweet):
		assert self.tokenizer is not None
		tokens = self.tokenizer(tweet.raw_text)
		V = self.vocab()
		return map(lambda t: V.get(t), tokens)
	def tokenize(self, tweet):
		return self.tokenizer(tweet.raw_text)




