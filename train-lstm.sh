#!/usr/bin/env bash

root="$PWD"
dataRoot="/home/kate/data/processed_tweets"

infile="$dataRoot/tweets.tsv"
vocab="$dataRoot/tweets.tsv.vocab"
testfile="$dataRoot/tweets.tsv.test"

logdir="$root/lstm_result"
mkdir -pv $logdir

python $root/semeval/lstm_words.py \
--tweet-file $infile \
--vocab $vocab \
--log-path $logdir \
--test-file $testfile

