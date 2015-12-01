#!/bin/bash

projRoot="/home/kate/Dropbox/F15/deep-learning/semeval16"
dataRoot="/home/kate/data/tweets"

#tweetFile="$dataRoot/training.1600000.processed.noemoticon.csv"
tweetFile="$dataRoot/training.small.csv"
outputFile="$projRoot/test/test.data"
testFile="$dataRoot/testdata.manual.2009.06.14.csv"
stopwords="$projRoot/lexica/stopwords.txt"

filterBelow=10
filterAbove=0.8

python process_1.6M_tweets.py \
--tweet-file $tweetFile \
--output-file $outputFile \
--test-file $testFile \
--stop-words $stopwords \
--filter-below $filterBelow \
--filter-above $filterAbove

