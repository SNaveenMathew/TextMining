# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:09:49 2017

@author: naveen.nathan
"""

from util import run_treetagger, postprocess_tag#, run_spacy
#from nltk.tag import StanfordPOSTagger
#from tokenization import tokenize_stanford
# from nltk import pos_tag
# from json import load
#pos_cfg = load(open("pos_tagger.cfg"))
#model = pos_cfg["tagger"]
#jar = pos_cfg["postagger.jar"]
#pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

# Available options:
# 1) Stanford
# 2) Treetagger (preferred)
# 3) spaCy

#def run_stanford_pos_tag(text = None, tokens = None, get_lemma = False, in_type = "tokens"):
#    if in_type == "text":
#        tokens = tokenize_stanford(text, get_lemma)
#        text = pos_tagger.tag(tokens)
#        return text
#    else:
#        text = pos_tagger.tag(tokens)
#        return text

# Purpose: Create POS tags using TreeTagger
# Input: String, Boolean (lemma required or not)
# Output: Tuple (string (lemma if required), POS)
def run_treetagger_pos_tag_text(text, get_lemma = False):
    try:
        try:
            text = run_treetagger(text)
            text = [postprocess_tag(s) for s in text]
            return text
        except:
            return text[0][0]
    except:
        return [('', '')]

#def run_spacy_pos_tag(text, get_lemma = False):
#    s = run_spacy(text)
#    lis = []
#    for word in s:
#        if get_lemma:
#            lis.append((word.text, word.lemma_))
#        else:
#            lis.append((word.text, word.tag_))
#    return lis
#

# Purpose: Lemmatizes a list of strings
# Input: List of strings
# Output: List of run_treetagger_pos_tag_text (tuple)
def run_treetagger_pos_tag_list(lis):
    ret = [run_treetagger_pos_tag_text(string) for string in lis]
    return ret

# def nltk_pos_tag(tokens):
#     return pos_tag(tokens)

