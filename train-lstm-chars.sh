#!/usr/bin/env bash

root="$PWD"
dataRoot="$root/CHAR_DATA/big_ascii"

infile="$dataRoot/tweets.chars.tsv.small"
vocab="$root/CHAR_DATA/ascii_vocab.pkl"
testfile="$dataRoot/tweets.chars.tsv.test"

logdir="$root/char_lstm_result_ascii"
mkdir -pv $logdir

THEANO_FLAGS=device=cpu,floatX=float32 python $root/semeval/lstm_chars.py \
--tweet-file $infile \
--vocab $vocab \
--log-path $logdir \
--test-file $testfile
