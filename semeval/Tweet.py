class Tweet:
	def __init__(self, id, label, raw_text):
		self.id = id
		self.label = label
		self.raw_text = raw_text
	def __repr__(self):
		return "Tweet(%s\t%s\t%s)" % (self.label, self.raw_text, self.id)

def load_from_tsv(filename):
	tweets = []
	with open(filename, "r") as f:
		for line in f.readlines():
			parts = line.split("\t")
			if len(parts) == 3:
				tweets.append(Tweet(parts[0], parts[1], parts[2].strip("\n")))
			else:
				print("bad line: %s" % line)
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
	print("loaded %d tweets total from file %s; %d failed to download properly." % (len(tweets), filename, errors))
	return tweets_clean