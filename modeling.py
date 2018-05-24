# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:17:17 2017

@author: naveen.nathan
"""

# Models:
# 1) Run word2vec on input text file
# 2) Combine unigrams to make meaningful bigrams. Then combine to make trigrams

# Purpose: To run word2vec model
# Input: Text file path
# Output: word2vec model object
def run_word2vec_model(text_file):
    from gensim.models.word2vec import LineSentence, Word2Vec
    sentences = LineSentence(text_file)
    model = Word2Vec(sentences, sg=1, workers=5, size=100, min_count=2, window=5)
    # model.build_vocab(sentences)
    model.train(sentences, total_examples = model.corpus_count, epochs = 5)
#    model.save("sample_model.w2v.bin")
    return model

# Purpose: Creates meaningful bigrams and trigrams from tokens
# Input: List of tokens (unigrams)
# Output: List (unigrams, bigrams (unigram_unigram) and trigrams(unigram_unigram_unigram))
def apply_bigram_trigram_model(unigrams):
    from gensim.models.phrases import Phrases , Phraser
    phrases = Phrases(unigrams)
    bigram = Phraser(phrases)
    trigram = Phrases(bigram[unigrams])
    trigram = Phraser(trigram)
    return list(trigram[bigram[unigrams]])

# Purpose: Creates topics based on LDA model
# Input: Input text file
# Output: LDA model object
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
    lda = LdaModel(corpus = corpus, id2word = dictionary, passes=1)
    return (lda, corpus, dictionary)

# Purpose: Creates Logistic Regression classification model
# Input: DataFrame of input columns and output column
# Output: Logistic Regression model object
def build_logistic_regression(df, outcome):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty = 'l1')
    model.fit(X = df.drop(outcome, axis=1), y = df[outcome])
    return model

# Purpose: Runs affinity propogation for clustering based on given m x m distance matrix
# Input: m x m square matrix of distances
# Output: Affinity propogation model object. aff.labels_ gives cluster labels
def run_aff_prop_with_distances(distances):
    from sklearn.cluster import AffinityPropagation
    aff = AffinityPropagation(max_iter = 1000, affinity = 'precomputed')
    aff.fit(distances)
    return aff

# Purpose: Runs kmeans clustering
# Input: DataFrame with required variables
# Output: DataFrame with cluster ID in 'cluster' column
def run_kmeans(data, outfile_prefix = ""):
    from sklearn.cluster import KMeans
    from pickle import dump
    from pandas import DataFrame
    n_cluster = optimal_k_silhouette(data, [i+2 for i in range(9)])
    kmeans = KMeans(init = 'k-means++', n_clusters =  n_cluster, n_init = 10)
    dump(kmeans, open(outfile_prefix + "kmeans_model.pkl", "wb"))
    cluster_labels = kmeans.fit_predict(data)
    data = DataFrame(data)
    data['cluster'] = cluster_labels
    return data

# Purpose: Silhouette criteria for automatic selection of number of clusters
# Input: DataFrame and range of number of clusters
# Output: Int (Optimal number of clusters)
def optimal_k_silhouette(data, range_n_clusters):
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    range_n_clusters = range_n_clusters
    silhouette_max = -1
    optimum_cluster = range_n_clusters[0]
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters = n_clusters, random_state = 10, n_init = 10)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        if silhouette_max < silhouette_avg :
            silhouette_max = silhouette_avg
            optimum_cluster = n_clusters
    return optimum_cluster

