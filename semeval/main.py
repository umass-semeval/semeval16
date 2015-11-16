import sys
from Tweet import *
from utils import *
from baseline import *

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
	corpus_stats(tweets)
	run_baseline(tweets)

if __name__ == "__main__":
	main(sys.argv[1:])

