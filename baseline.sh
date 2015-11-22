#!/bin/bash

modelType="word2vec_logistic_regression"
#modelType="bow_logistic_regression"
#modelType="naive_bayes"

stopwords="$root/lexica/stopwords.txt"
word2vec="/home/kate/research/word2vec/trunk/vectors.bin"

root=`pwd`
echo "root: $root"

subtaskID="b"
trainData="$root/data/subtask-B/train.tsv"
devData="$root/data/subtask-B/dev.tsv"
python $root/semeval/baseline.py \
--subtask-id $subtaskID \
--train-file $trainData \
--dev-file $devData \
--model-type $modelType \
--stopwords $stopwords \
--word2vec-model $word2vec

#subtaskID="a"
#trainData="$root/data/subtask-A/train.tsv"
#devData="$root/data/subtask-A/dev.tsv"
#testData="$root/data/subtask-A/test.tsv"
#python $root/semeval/baseline.py \
#--subtask-id $subtaskID \
#--train-file $trainData \
#--dev-file $devData \
#--test-file $testData \
#--model-type $modelType \
#--stopwords $stopwords \
#--word2vec-model $word2vec


