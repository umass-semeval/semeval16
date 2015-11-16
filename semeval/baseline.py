from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import utils

def classmap(tweets):
	classes = get_classes(tweets)
	class_map = {}
	i = 0
	for c in classes:
		class_map[c] = i
		i += 1
	return (classes, class_map)

def get_classes(tweets):
	return list(set(map(lambda x: x.label, tweets)))

def extract_features(tweets, count_vec, tt, is_test):
	strings = map(lambda x: x.raw_text, tweets)
	x_tfidf = None
	if not is_test:
		x_counts = count_vec.fit_transform(strings)
		x_tfidf = tt.fit_transform(x_counts)
	else:
		x_counts = count_vec.transform(strings)
		x_tfidf = tt.transform(x_counts)
	return x_tfidf

def accuracy(tweets, prediction, classes, extra="", verbose=False):
	ncorrect = 0.0
	total = 0.0
	for tweet, category in zip(tweets, prediction):
		predicted = classes[category]
		true_class = tweet.label
		correct = ""
		if predicted == true_class:
			correct = ""
			ncorrect += 1.0
		else:
			correct = "*"
		total += 1.0
		if verbose:
			print("%s\t%s %s : %s" % (correct, true_class, predicted, tweet.raw_text))
	accuracy = ncorrect/total	
	print("%s accuracy: %f (%d/%d)" % (extra, accuracy, int(ncorrect), int(total)))
	return accuracy


def run_baseline(full_tweets):
	train,test = utils.split_data(full_tweets)
	classes, cmap = classmap(train)
	count_vec = CountVectorizer()
	tt = TfidfTransformer()	
	x_train = extract_features(train, count_vec, tt, False)
	y_train = map(lambda x: cmap[x.label], train)
	clf = MultinomialNB()
	clf.fit(x_train, y_train)
	prediction = clf.predict(x_train)
	accuracy(train, prediction, classes, extra="train")
	x_test = extract_features(test, count_vec, tt, True)
	y_test = map(lambda x: cmap[x.label], test)
	test_prediction = clf.predict(x_test)
	accuracy(test, test_prediction, classes, extra="test")




