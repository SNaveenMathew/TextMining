# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:22:02 2017

@author: naveen.nathan
"""

from util import read_file#, flatten_list_of_list#, clean_sentences
from util import pick_first_language, is_english_wp_p#, spell_correct_tokens
from util import detect_language#, clean_strings
from util_spark import remove_stopwords_spark
from tokenization import tokenize_sentence_nltk, tokenize_spark
from modeling import run_word2vec_model_pyspark
from langdetect import DetectorFactory
from pandas import Series

with open("setupPySpark.py", "r") as setup_file:
    exec(setup_file.read())

from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession

spark = SparkSession(sc)
DetectorFactory.seed = 0

# Testing language
in_file = open("in_file.cfg").read()
in_file = in_file.split("\n")
label = in_file[3]
column = in_file[2]
in_type = in_file[1]
in_file = in_file[0]
strings = read_file(in_file, in_type = in_type)
if(in_type == "text"):
    strings = tokenize_sentence_nltk(strings)
    strings = Series(strings)
else:
    if(label in strings.columns):
        labels = strings[label]

languages = strings[column].apply(detect_language)

# Picking the language with highest probability
first_language = languages.apply(pick_first_language)

# Keeping only English text
english_only = first_language.apply(is_english_wp_p)
strings = strings[english_only]
if(label in strings.columns):
    labels = labels[english_only].tolist()
sentenceDataFrame = spark.createDataFrame(strings)
sentenceDataFrame.withColumn(column, regexp_replace(column, pattern = '^(nan )*',
                                                    replacement = ''))
tokenized = tokenize_spark(sentenceDataFrame, "text", "words")
stopwords_removed = remove_stopwords_spark(tokenized, "words")
model = run_word2vec_model_pyspark(stopwords_removed)
