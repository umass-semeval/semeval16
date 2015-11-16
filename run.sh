#!/bin/bash

root="/home/kate/Dropbox/F15/deep-learning/semeval16"
data="$root/data/subtask-A/train.tsv"

python $root/semeval/main.py --train-file $data --stopwords $root/lexica/stopwords.txt
