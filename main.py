# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 18:28:47 2017

@author: naveen.nathan
"""

# Loading required libraries and initializing
from langdetect import DetectorFactory
from os import chdir
from pandas import DataFrame
from string import punctuation

# Loading custom defined functions
wd = open('wd.cfg').read()
chdir(wd)
from tokenization import tokenize_sentence_nltk#, tokenize_treetagger
from util import read_file, flatten_list_of_list#, clean_sentences
from util import pick_first_language, is_english_wp_p, spell_correct_tokens
from util import detect_language, clean_strings
from pos_tagging import run_treetagger_pos_tag_text
from modeling import run_word2vec_model, apply_bigram_trigram_model
from modeling import run_lda_topic_model, build_logistic_regression
from visualizing import visualize_word2vec_model

# Current spell corrector is based on spell function in autocorrect package
# WORDS = Counter(tokenize_stanford(open('big.txt').read()))

DetectorFactory.seed = 0

# Testing language
in_file = open("in_file.cfg").read()
in_file = in_file.split("\n")
col = in_file[2]
in_type = in_file[1]
in_file = in_file[0]
strings = read_file(in_file, in_type = in_type, col = col)
if(in_type == "text"):
    strings = tokenize_sentence_nltk(strings)
    strings = DataFrame(strings)[0]

strings = strings.apply(clean_strings)
languages = strings.apply(detect_language)

# Picking the language with highest probability
first_language = languages.apply(pick_first_language)

# Keeping only English text
english_only = first_language.apply(is_english_wp_p)
strings = strings[english_only]

# Processing English sentences:
# 1) Splitting sentences
sentences = strings.apply(tokenize_sentence_nltk)
sentences = flatten_list_of_list(sentences)
# 2) Run part-of-speech tagging on clean sentences
sentences = DataFrame(sentences)[0]
pos = sentences.apply(run_treetagger_pos_tag_text).apply(DataFrame)
# 3) Spell correct - currently correct only disjoint words
lengths = pos.apply(len)
inc_sentences = sentences[lengths == 0]
pos = pos[lengths > 0]
sentence_tokens = pos.apply(spell_correct_tokens)
# 4) Combine tokens to form bigrams and trigrams
# sentence_tokens = sentences.apply(tokenize_treetagger)
trigrams = apply_bigram_trigram_model(sentence_tokens)
# 5) Form the sentence back from tokens
sentences1 = ["".join([" "+i if not i.startswith("'") and i not in punctuation
                       else i for i in tokens]).strip() for tokens in trigrams]

# Write clean text to text file - one line per sentence
out_file = open("sample.txt", "w")
for sent in sentences1:
    out_file.write(sent.lower().replace("( ", "(").replace(" )", ")").replace("replaced-dns ", "")+"\n")

out_file.close()

# Run word2vec model and store word representations
model = run_word2vec_model("sample.txt")

visualize_word2vec_model(model)