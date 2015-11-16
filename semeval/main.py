import sys
from Tweet import *
from utils import *
from Preprocessor import *
from logreg import *

def corpus_stats(tweets):
	classes = set(map(lambda x: x.label, tweets))
	print("classes: %s" % ",".join(list(classes)))
	for c in classes:
		examples = filter(lambda x: x.label == c, tweets)
		print("class %s: %d tweets total" % (c, len(examples)))

def main(*args):
	filename = args[0][0]
	tweets = load_from_tsv(filename)
	for t in tweets[:10]:
		print(t)
	train_data, test_data = split_data(tweets)
	preprocessor = Preprocessor()
	preprocessor.build_vocab(train_data)
	train_x = map(lambda t: preprocessor.preprocess_constant_len(t, 50), train_data)
	classes = list(set(map(lambda t: t.label, train_data)))
	train_y = np.array(map(lambda t: classes.index(t.label), train_data))
	assert len(train_x) == len(train_y)

	ndims = train_x[0].shape[0]
	nclasses = len(classes)
	n = len(train_x)

	print("ndims: %d, nclasses: %d" % (ndims, nclasses))
	print("n: %d" % n)

	X = np.mat(train_x)
	print(X.shape)
	print(train_y.shape)

	do_logreg_ova(X, train_y, ndims, nclasses, n)
	# do_logreg()



if __name__ == "__main__":
	main(sys.argv[1:])

