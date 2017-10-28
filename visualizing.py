# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:55:54 2017

@author: naveen.nathan
"""

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_word2vec_model(word2vec_model):
    X = word2vec_model[word2vec_model.wv.vocab]
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    vocabulary = word2vec_model.wv.vocab
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    for label, x, y in zip(vocabulary, X_tsne[:, 0], X_tsne[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    return(plt.show())
