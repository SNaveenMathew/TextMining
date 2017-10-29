# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:33:25 2017

@author: naveen.nathan
"""

from treetaggerwrapper import TreeTagger, make_tags
#from en_core_web_md import load
from os import environ
from pandas import read_csv, DataFrame
from itertools import chain
from re import sub
from autocorrect import spell
from langdetect import detect_langs
import langdetect
from autocorrect.nlp_parser import NLP_WORDS
from nltk.corpus import stopwords
from string import punctuation

# Initializating global variables
# Initialization for:
# 1) Treetagger
# 2) Spacy
treetagger_home = open('treetagger.cfg').read()
environ["TREETAGGER_HOME"] = treetagger_home
tagger = TreeTagger(TAGLANG = 'en')
NLP_WORDS = set([word.lower() for word in NLP_WORDS])
english_stopwords = set(stopwords.words('english'))
puncts = set(punctuation)
# Commenting spaCy to reduce initialization overhead
#nlp = load()

# Other utilities:
# 1) Read text file
# 2) Flatten a double-list into list
# 3) Clean beginning of sentences
# 4) Pick language with highest probability from set of languages
# 5) Check whether language is English with prabability > p (default = 0.5)
# 6) Spell correct (in progress)

def run_treetagger(text):
    s = tagger.tag_text(text.lower())
    s = make_tags(s)
    return(s)

#def run_spacy(text):
#    doc = nlp(text.lower())
#    return(doc)
#
def read_file(file, in_type = "csv"):
    if(in_type == "csv"):
        return(read_csv(file, encoding = "latin1"))
    else:
        text = open(file, 'r').read()
        return(text)

def flatten_list_of_list(list_of_list):
    return(list(chain.from_iterable(list_of_list)))

def clean_sentences(sentences):
    return([clean_strings(string) for string in sentences])

def clean_strings(string):
    return(sub(pattern = "^(nan )*", repl = "", string = string))

def pick_first_language(langs):
    if(langs!=None):
        return(langs[0])
    else:
        return(langdetect.language.Language(lang = "NA", prob = 0))

def is_english_wp_p(langs, p = 0.5):
    return(langs.lang == "en" and langs.prob > p)

def diffs(index, tokens):
    return([tokens[index], tokens[index+1]])

def merge_words(words):
    return(words[0]+words[1])

def correct_tokens(tokens, wrong_corrected, combine_check):
    final_tokens = []
    i = 0
    j = 0
    while i<len(tokens):
        if i!=combine_check[j]:
            final_tokens.append(tokens[i])
        else:
            final_tokens.append(wrong_corrected[combine_check[j]])
            j = j+1
            i = i+1
        i = i+1
    return(DataFrame(final_tokens)[0])

def lower(text):
    return(text.lower())

def check_spell(row):
    if(len(row[0])==1 or row[1] in [")", "(", "''", "PP$", ",", ":", '``']):
        return(row[0])
    else:
        return(spell(row[0]))

def is_in_words(word):
    return(word in NLP_WORDS)

# This is yet to  be developed fully. It currently returns the tokens as they are
def spell_correct_tokens(pos):
    # This only merges 2 consecutive words & checks if they are both incorrectly spelled
    try:
        tokens = pos[pos[1]!="SENT"]
        updated_tokens = tokens.apply(check_spell, axis = 1).apply(lower)
        same = updated_tokens != tokens[0]
        diff = DataFrame(same.index.values)[0][same]
        if(len(diff)>0):
            wrong = diff.apply(diffs, args = (tokens, ))
            wrong_merge = wrong.apply(merge_words)
            wrong_corrected = wrong_merge.apply(spell).apply(lower)
            same1 = wrong_corrected == wrong_merge
            combine_check = diff[same1]
            wrong_corrected = wrong_corrected[same1]
            same2 = wrong_corrected.apply(is_in_words)
            wrong_corrected = wrong_corrected[same2]
            combine_check = combine_check[same2]
            if(len(wrong_corrected)>0):
                tokens = correct_tokens(tokens[0], wrong_corrected, combine_check)
            else:
                tokens = tokens[0]
        else:
            tokens = tokens[0]
        if(pos[1][len(pos)-1] == "SENT"):
            tokens = tokens.append(DataFrame([pos[0][len(pos)-1]]),
                                   ignore_index=True)
        return(tokens.tolist())
    except:
        return(pos[0].tolist())

def is_not_none(row):
    return(row!=None)

#def spell_correct_pos(pos):
#    try:
#        tokens = spell_correct_tokens(pos)[0].tolist()
#        return(tokens)
#    except:
#        return(pos[0].tolist())
#
def process_NotTag(not_tag):
    text = not_tag.split('"')
    return(text[1])

def detect_language(text):
    try:
        return(detect_langs(text))
    except:
        return(None)

def remove_stopwords(tokens):
    tokens = [token for token in tokens if token not in english_stopwords]
    return(tokens)

def remove_punctuations(tokens):
    tokens = [token for token in tokens if token not in puncts]
    return(tokens)
