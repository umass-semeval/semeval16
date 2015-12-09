#!/usr/bin/env bash

root="$PWD"
dataRoot="/home/kate/data/processed_tweets"

infile="$dataRoot/tweets.tsv.small"
vocab="$dataRoot/tweets.tsv.vocab"
testfile="$dataRoot/tweets.tsv.test"

embeddings="/home/kate/data/SSWE/sswe-u.txt.w2v"

logdir="$root/lstm_result"
mkdir -pv $logdir

#python $root/semeval/lstm_words.py \
#--tweet-file $infile \
#--vocab $vocab \
#--log-path $logdir \
#--test-file $testfile

python $root/semeval/lstm_words.py \
--tweet-file $infile \
--vocab $vocab \
--log-path $logdir \
--test-file $testfile \
--embeddings-file $embeddings