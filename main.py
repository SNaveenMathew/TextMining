# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 18:28:47 2017

@author: naveen.nathan
"""

# Loading required libraries and initializing
from langdetect import DetectorFactory
from pandas import Series, DataFrame
from string import punctuation
#from numpy import zeros
#from pickle import dump
#from os import chdir

# Loading custom defined functions
#wd = open('wd.cfg').read()
#chdir(wd)
from tokenization import tokenize_sentence_nltk#, tokenize_treetagger
from util import read_file, flatten_list_of_list, read_folder#, clean_sentences
from util import pick_first_language, is_english_wp_p, spell_correct_tokens
from util import detect_language, clean_strings, run_treetagger
from util import filter_data, filter_senders, filter_recipients
from pos_tagging import run_treetagger_pos_tag_text
from modeling import apply_bigram_trigram_model
from lemmatization import lemmatize_treetagger
#from modeling import run_word2vec_model, run_lda_topic_model, build_logistic_regression
#from visualizing import visualize_word2vec_model

# Current spell corrector is based on spell function in autocorrect package
# WORDS = Counter(tokenize_stanford(open('big.txt').read()))

DetectorFactory.seed = 0

# Testing language
in_file = open("in_file.cfg").read()
in_file = in_file.split("\n")
file_folder = in_file[4]
label = in_file[3]
col = in_file[2]
in_type = in_file[1]
in_file = in_file[0]
if file_folder == "file":
    strings = read_file(in_file, in_type = in_type)
    if(in_type == "text"):
        strings = tokenize_sentence_nltk(strings)
        strings = DataFrame(strings)[0]
    elif(in_type == "html"):
        timestamp = strings[2]
        meta_data = strings[1]
        strings = strings[0]
        strings[label] = meta_data["Comment"]
        labels = strings[label]
        strings = strings[col]
    else:
        if(label in strings.columns):
            labels = strings[label]
        strings = strings[col]
else:
    strings = read_folder(in_file)

# Some deduplication to be done here to keep the remaining steps same
strings = filter_data(strings)
strings = filter_senders(strings)
strings = filter_recipients(strings)

strings = strings.apply(clean_strings)
languages = strings.apply(detect_language)

# Picking the language with highest probability
first_language = languages.apply(pick_first_language)

# Keeping only English text
english_only = first_language.apply(is_english_wp_p)
strings = strings[english_only]
# labels = labels[english_only].tolist()

# Processing English sentences:
# 1) Splitting sentences
sentences = strings.apply(tokenize_sentence_nltk)
lengths = sentences.apply(len).tolist()
# new_labels = []
# for i in range(len(lengths)):
#     for j in range(lengths[i]):
#         new_labels.append(labels[i])
# new_labels = Series(new_labels)
sentences1 = flatten_list_of_list(sentences)
# 2) Run part-of-speech tagging on clean sentences
sentences1 = Series(sentences1)
pos = sentences1.apply(run_treetagger_pos_tag_text).apply(DataFrame)
# 3) Spell correct - currently correct only disjoint words
lengths = pos.apply(len)
inc_sentences = sentences1[lengths == 0]
# inc_labels = new_labels[lengths == 0]
pos = pos[lengths > 0]
# labels = new_labels[lengths > 0]
sentence_tokens = pos.apply(spell_correct_tokens)
# 4) Combine tokens to form bigrams and trigrams
# sentence_tokens = sentences.apply(tokenize_treetagger)
trigrams = apply_bigram_trigram_model(sentence_tokens)
# 5) Form the sentence back from tokens
sentences1 = ["".join([" "+lemmatize_treetagger(run_treetagger(i)) if not i.startswith("'") and i not in punctuation
                       else i for i in tokens]).strip() for tokens in trigrams]
sentences1 = sentences1 + inc_sentences.tolist()
# labels = labels.tolist() + inc_labels.tolist()

# 6) Write clean text to text file - one line per sentence
out_file = open("sample.txt", "w")
for sent in sentences1:
    out_file.write(sent.lower().replace("( ", "(").replace(" )", ")").replace("replaced-dns ", "")+"\n")
out_file.close()

# 7) Run word2vec model and store word representations
#model = run_word2vec_model("sample.txt")
#model.wv.save_word2vec_format("big.w2v")

# 8) Visualizing the word2vec model
#visualize_word2vec_model(model)

# 9) Setting up the data for building logistic regression model
#df = zeros((len(sentences1), 100))
#for i, words in enumerate(trigrams):
#    for word in words:
#        try:
#            df[i] = df[i] + model[word]
#        except:
#            continue

#while(i<len(sentences1)):
#    i += 1
#
#df = DataFrame(df)
#df[label] = labels

# 10) Building and saving the logistic regression model with L1 penalty
#lr_model = build_logistic_regression(df, label)
#dump(lr_model, open("logistic_model.pkl", 'wb'))

# 11) Topic modeling (Optional)
#lda_model = run_lda_topic_model(text_file = "sample_cleaned.txt")
# Sample topic modeling output - Topic 1
#lda_model.print_topic(1)
