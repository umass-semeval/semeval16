#!/bin/bash

modelType="word2vec_logistic_regression"

root="/home/kate/Dropbox/F15/deep-learning/semeval16"
trainData="$root/data/subtask-A/train.tsv"
devData="$root/data/subtask-A/dev.tsv"
testData="$root/data/subtask-A/test.tsv"

stopwords="$root/lexica/stopwords.txt"
word2vec="/home/kate/research/word2vec/trunk/vectors.bin"

python $root/semeval/baseline.py \
--train-file $trainData --dev-file $devData --test-file $testData \
--model-type $modelType --stopwords $stopwords --word2vec-model $word2vec

