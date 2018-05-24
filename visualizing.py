# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:55:54 2017

@author: naveen.nathan
"""

# Purpose: Visualize a word2vec model
# Input: Gensim word2vec model, color series
# Output: matplotlib plot with color
def visualize_word2vec_model(word2vec_model, color = None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    X = word2vec_model[word2vec_model.wv.vocab]
    tsne = TSNE(n_components=2, random_state = 1)
    X_tsne = tsne.fit_transform(X)
    vocabulary = word2vec_model.wv.vocab
    if color is None:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = color)
    for label, x, y in zip(vocabulary, X_tsne[:, 0], X_tsne[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    return plt.show()

def visualize_lda_topics(lda_model, corpus, dictionary):
    import pyLDAvis
    import pyLDAvis.gensim
    pyLDAvis.enable_notebook()
    return pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
