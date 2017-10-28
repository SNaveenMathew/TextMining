# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:52:31 2017

@author: naveen.nathan
"""

#from nltk.tokenize.stanford import StanfordTokenizer
from util import run_treetagger, flatten_list_of_list, process_NotTag
from lemmatization import lemmatize_treetagger
from nltk.tokenize import sent_tokenize
#from os import environ
import treetaggerwrapper

# Initialization
# Classpath is required to initialize Stanford POS Tagger
#classpath = open("pos_tagger.cfg").read()
#classpath = classpath.split("\n")[0]
#environ["CLASSPATH"] = classpath
#s = StanfordTokenizer()

# Word tokenization options:
# 1) Treetagger tokenizer (preferred)
# 2) Stanford tokenizer
# 3) spaCy tokenizer (close to Treetagger)

# Sentence tokenization options:
# 1) NLTK sentence tokenizer

#def tokenize_stanford(text, get_lemma = False):
#    text = s.tokenize(text.lower())
#    return text
#
def tokenize_treetagger(text, get_lemma = False):
    s = run_treetagger(text)
    if(get_lemma):
        s = [lemmatize_treetagger(tag) for tag in s]
        return(s)
    else:
        s = [tag[0] if type(tag)!=treetaggerwrapper.NotTag else process_NotTag(tag[0]) for tag in s]
        return(s)

#def tokenize_spacy(text, get_lemma = False):
#    s = run_spacy(text)
#    lis = []
#    for word in s:
#        if get_lemma:
#            lis.append(word.lemma_)
#        else:
#            lis.append(word.text)
#    return(lis)
#
def tokenize_sentence_nltk(text, get_lemma = False):
    text = text.replace("\n", " ").replace("\xa0", " ")
    text = sent_tokenize(text)
    text = [sent.split("(.[A-Z])") for sent in text]
    text = flatten_list_of_list(text)
    return(text)
