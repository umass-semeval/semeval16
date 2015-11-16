def split_data(tweets, train_portion=0.8):
	n = float(len(tweets))
	trainp = int(train_portion * n)
	train = tweets[:trainp]
	test = tweets[trainp:]
	return (train,test)