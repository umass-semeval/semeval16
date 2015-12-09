#!/bin/bash

root="$PWD"
dataRoot="/home/kate/data/tweets"
outDir="/home/kate/data/processed_tweets"

mkdir -pv $outDir

infile="$dataRoot/training.1600000.processed.noemoticon.csv"
testfile="$dataRoot/testdata.manual.2009.06.14.csv"

outfile="$outDir/tweets.tsv"

stopwords="$dataRoot/lexica/stopwords.txt"
filter_below=30
filter_above=0.8

python process_tweets_ks.py \
--tweet-file $infile \
--test-file $testfile \
--output-file $outfile \
--stop-words $stopwords \
--filter-below $filter_below \
--filter-above $filter_above

