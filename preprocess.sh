#!/bin/bash

function checkfile() {
filename=$1
[ -f $filename ] && echo "$filename exists" || echo "$filename does not exist"
}

function checkdir() {
filename=$1
[ -d $filename ] && echo "dir $filename exists" || echo "dir $filename does not exist"
}

root="$PWD"
dataRoot="/home/kate/data/tweets"
outDir="/home/kate/data/processed_tweets"

mkdir -pv $outDir

infile="$dataRoot/training.1600000.processed.noemoticon.csv"
testfile="$dataRoot/testdata.manual.2009.06.14.csv"

outfile="$outDir/tweets.tsv"

stopwords="/home/kate/data/lexica/stopwords.txt"
filter_below=30
filter_above=0.8

checkdir $dataRoot
checkfile $infile
checkfile $testfile
checkfile $stopwords
checkdir $outDir

python process_tweets_ks.py \
--tweet-file $infile \
--test-file $testfile \
--output-file $outfile \
--stop-words $stopwords \
--filter-below $filter_below \
--filter-above $filter_above

