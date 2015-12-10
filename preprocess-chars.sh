#!/usr/bin/env bash

#function checkfile() {
#filename=$1
#[ -f $filename ] && echo "$filename exists" || echo "$filename does not exist"
#}
#
#function checkdir() {
#filename=$1
#[ -d $filename ] && echo "dir $filename exists" || echo "dir $filename does not exist"
#}

if [ -z $1 ]
then
echo "PROCESS SEMEVAL"
root="$PWD"
dataRoot="/home/kate/F15/semeval16/data/subtask-A"
outDir="$root/CHAR_DATA/semeval"
mkdir -pv $outDir
trainfile="$dataRoot/train.tsv"
devfile="$dataRoot/dev.tsv"
testfile="$dataRoot/test.tsv"
labelMap="$root/CHAR_DATA/labels.pkl"
vocab="$root/CHAR_DATA/vocab.pkl"
checkfile $vocab
checkfile $labelMap
checkdir $outDir
python semeval/preprocess_chars.py \
--semeval "true" \
--output-dir $outDir \
--tweet-file $trainfile \
--test-file $testfile \
--dev-file $devfile \
--label-map $labelMap \
--vocab-file $vocab

else
#
# PROCESS 1.6M
#
echo "PROCESS 1.6M"
root="$PWD"
dataRoot="/home/kate/data/tweets"
outDir="$root/CHAR_DATA/big"
mkdir -pv $outDir
infile="$dataRoot/training.1600000.processed.noemoticon.csv"
testfile="$dataRoot/testdata.manual.2009.06.14.csv"
labelMap="$root/CHAR_DATA/labels.pkl"
outfile="$outDir/tweets.chars.tsv"
checkdir $dataRoot
checkfile $infile
checkfile $testfile
checkdir $outDir
python semeval/preprocess_chars.py \
--tweet-file $infile \
--test-file $testfile \
--output-file $outfile \
--label-map $labelMap

fi