from pyspark.ml.feature import StopWordsRemover

def remove_stopwords_spark(tokenized, in_column, out_column = None):
    if out_column is not None:
        remover = StopWordsRemover(inputCol = in_column, outputCol = out_column)
    else:
        remover = StopWordsRemover(inputCol = in_column)
    stopwords_removed = remover.transform(tokenized)
    return stopwords_removed
