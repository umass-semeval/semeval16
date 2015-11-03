******************************************************
* SemEval-2016 Task 4: Sentiment Analysis on Twitter *
*                                                    *
*               TRAINING + DEV DATA                  *
*                                                    *
* http://alt.qcri.org/semeval2016/task4/             *
* semevaltweet@googlegroups.com                      *
*                                                    *
******************************************************


TRAINING + DEV dataset for SemEval-2016 Task 4

Version 1.1: October 31, 2015


Task organizers:

* Preslav Nakov, Qatar Computing Research Institute, HBKU
* Alan Ritter, The Ohio State University
* Sara Rosenthal, Columbia University
* Fabrizio Sebastiani, Qatar Computing Research Institute, HBKU
* Veselin Stoyanov, Facebook


LIST OF VERSIONS

  v1.1 [2015/10/31]: swapped the first columns of the datasets for subtasks B, C, D, E due to an issue with the download script (otherwise the data is the same)

  v1.0 [2015/10/15]: initial distribution of the data


NOTES

1. Please note that by downloading the Twitter data you agree to abide by the Twitter terms of service (https://twitter.com/tos), and in particular you agree not to redistribute the data and to delete tweets that are marked deleted in the future.

2. The distribution consists of a set of Twitter status IDs with annotations for Subtasks A, B, C, D, and E: topic polarity and trends toward a topic. There are exactly 100 tweets provided per topic and a total of 100 topics. You should use the downloading script to obtain the corresponding tweets: https://github.com/aritter/twitter_download

3. The "neutral" label in the annotations stands for objective_OR_neutral.


FILES

train/100_topics_100_tweets.sentence-three-point.subtask-A.train.txt -- training input for subtask A
train/100_topics_100_tweets.topic-two-point.subtask-BD.train.txt -- training input for subtasks B and D
train/100_topics_100_tweets.topic-five-point.subtask-CE.train.txt -- training input for subtasks C and E

dev/100_topics_100_tweets.sentence-three-point.subtask-A.dev.txt -- dev input for subtask A
dev/100_topics_100_tweets.topic-two-point.subtask-BD.dev.txt -- dev input for subtasks B and D
dev/100_topics_100_tweets.topic-five-point.subtask-CE.dev.txt -- dev input for subtasks C and E


INPUT DATA FORMAT


-----------------------SUBTASK A-----------------------------------------

The format for the training/dev file is as follows:

	id<TAB>label

where "label" can be 'positive', 'neutral' or 'negative'.


-----------------------SUBTASKS B,D--------------------------------------

The format for the training/dev file is as follows:

	id<TAB>topic<TAB>label

where "label" can be 'positive' or 'negative' (note: no 'neutral'!).


-----------------------SUBTASKS C,E--------------------------------------

The format for the training/dev file is as follows:

	id<TAB>topic<TAB>label

where "label" can be -2, -1, 0, 1, or 2,
corresponding to "strongly negative", "negative", "negative or neutral", "positive", and "strongly positive".



LICENSE

The accompanying dataset is released under a Creative Commons Attribution 3.0 Unported License
(http://creativecommons.org/licenses/by/3.0/).


CITATION

You can cite the folowing paper when referring to the dataset:

@InProceedings{Rosenthal-EtAl:2015:SemEval,
  author    = {Sara Rosenthal and Alan Ritter and Veselin Stoyanov and Svetlana Kiritchenko and Saif Mohammad and Preslav Nakov},
  title     = {SemEval-2015 Task 10: Sentiment Analysis in Twitter},
  booktitle = {Proceedings of the 9th International Workshop on Semantic Evaluation (SemEval 2015)},
  year      = {2015},
  publisher = {Association for Computational Linguistics},
}


USEFUL LINKS:

Google group: semevaltweet@googlegroups.com
SemEval-2016 Task 4 website: http://alt.qcri.org/semeval2016/task4/
SemEval-2016 website: http://alt.qcri.org/semeval2016/
