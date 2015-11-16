from gensim.models import word2vec
from Corpus import *
import utils

def embed(tweets):
	train,test = utils.split(tweets)
	corpus = Corpus(train)
	all_sentences = map(lambda t: corpus.tokenize(t), tweets)
	train_sentences = map(lambda t: corpus.tokenize(t), train)
	test_sentences = map(lambda t: corpus.tokenize(t), test)
	model = word2vec.Word2Vec()
	model.build_vocab(all_sentences)
	model.train(train_sentences)
	return model
