# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:56:27 2017

@author: naveen.nathan
"""

import treetaggerwrapper
#from nltk.stem import WordNetLemmatizer

# Initializing
# Commenting WordNet lemmatizer to avoid initialization overhead
#wnl = WordNetLemmatizer()

# Available options:
# 1) Treetagger (preferred)
# 2) Wordnet
# 3) spaCy (in progress)

def lemmatize_treetagger(tag):
    length = len(tag)
    if length>1 and type(tag)==treetaggerwrapper.Tag:
        if tag[length-1]!="@card@":
            return tag[length-1]
        else:
            return tag[0]
    elif type(tag)==list:
        lemma = []
        for t in tag:
            lemma = lemma + [lemmatize_treetagger(t)]
        lemma = " ".join(lemma)
        return lemma
    else:
        return tag

#def lemmatize_wordnet(tokens):
#    return [wnl.lemmatize(s) for s in tokens]
#
#def lemmatize_spacy(tokens):
#    return tokens
