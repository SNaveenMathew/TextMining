# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:09:49 2017

@author: naveen.nathan
"""

from util import run_treetagger#, run_spacy
#from nltk.tag import StanfordPOSTagger
#from tokenization import tokenize_stanford
# from nltk import pos_tag
#pos_cfg = open("pos_tagger.cfg").read()
#pos_cfg = pos_cfg.split("\n")
#model = pos_cfg[1]
#jar = pos_cfg[0]
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

def run_treetagger_pos_tag_text(text, get_lemma = False):
    try:
        text = run_treetagger(text)
        if get_lemma:
            text = [(str(s[2]), str(s[1])) for s in text]
        else:
            text = [(str(s[0]), str(s[1])) for s in text]
        return text
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
def run_treetagger_pos_tag_list(lis):
    ret = [run_treetagger_pos_tag_text(string) for string in lis]
    return ret

# def nltk_pos_tag(tokens):
#     return pos_tag(tokens)

