# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:17:17 2017

@author: naveen.nathan
"""

from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.models.phrases import Phrases , Phraser
from gensim.models.ldamodel import LdaModel
from gensim.corpora.textcorpus import TextCorpus
import gensim.corpora.dictionary as dic
from tokenization import tokenize_treetagger
from util import remove_stopwords, remove_punctuations
from pandas import Series
from sklearn.linear_model import LogisticRegression

# Models:
# 1) Run word2vec on input text file
# 2) Combine unigrams to make meaningful bigrams. Then combine to make trigrams

def run_word2vec_model(text_file):
    sentences = LineSentence(text_file)
    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
#    model.save("sample_model.w2v.bin")
    return(model)

def apply_bigram_trigram_model(unigrams):
    phrases = Phrases(unigrams)
    bigram = Phraser(phrases)
    trigram = Phrases(bigram[unigrams])
    trigram = Phraser(trigram)
    return(list(trigram[bigram[unigrams]]))

def run_lda_topic_model(text_file):
    corpus = TextCorpus(text_file)
    text = open(text_file, 'r').read()
    text = text.split("\n")
    text = Series(text)
    text = text.apply(tokenize_treetagger)
    text = text.apply(remove_stopwords)
    text = text.apply(remove_punctuations)
    dictionary = dic.Dictionary(text)
    corpus = [dictionary.doc2bow(sent) for sent in text]
    lda = LdaModel(corpus = corpus, id2word = dictionary, passes=20)
    return(lda)

def build_logistic_regression(df, outcome):
    model = LogisticRegression(penalty = 'l1')
    model.fit(X = df.drop(outcome, axis=1), y = df[outcome])
    return(model)
