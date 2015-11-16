#!/bin/bash

root="/home/kate/Dropbox/F15/deep-learning/semeval16"
trainData="$root/data/subtask-A/train.tsv"
devData="$root/data/subtask-A/dev.tsv"

python $root/semeval/baseline.py --train-file $trainData --dev-file $devData

