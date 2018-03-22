# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:17:17 2017

@author: naveen.nathan
"""

#from pyspark.ml.feature import Word2Vec as w2v

# Models:
# 1) Run word2vec on input text file
# 2) Combine unigrams to make meaningful bigrams. Then combine to make trigrams

def run_word2vec_model(text_file):
    from gensim.models.word2vec import LineSentence, Word2Vec
    sentences = LineSentence(text_file)
    model = Word2Vec(sentences, sg=1, workers=5, size=100, min_count=2, window=5)
    # model.build_vocab(sentences)
    model.train(sentences, total_examples = model.corpus_count, epochs = 20)
#    model.save("sample_model.w2v.bin")
    return model

def apply_bigram_trigram_model(unigrams):
    from gensim.models.phrases import Phrases , Phraser
    phrases = Phrases(unigrams)
    bigram = Phraser(phrases)
    trigram = Phrases(bigram[unigrams])
    trigram = Phraser(trigram)
    return list(trigram[bigram[unigrams]])

def run_lda_topic_model(text_file):
    from gensim.models.ldamodel import LdaModel
    from gensim.corpora.textcorpus import TextCorpus
    import gensim.corpora.dictionary as dic
    from tokenization import tokenize_treetagger
    from util import remove_stopwords, remove_punctuations
    from pandas import Series
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
    return lda

def build_logistic_regression(df, outcome):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty = 'l1')
    model.fit(X = df.drop(outcome, axis=1), y = df[outcome])
    return model

#def run_word2vec_model_pyspark(documentDF, tokens, out_col = None):
#    if out_col is not None:
#        model = w2v(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
#    else:
#        model = w2v(vectorSize=3, minCount=0, inputCol="text")
#    result = model.fit(documentDF)
#    for row in result.collect():
#        text, vector = row
#        print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))
#    return result
