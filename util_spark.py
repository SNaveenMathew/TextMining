from pyspark.ml.feature import StopWordsRemover
from util import detect_language
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

detect_language_udf = udf(lambda x: detect_language(" ".join(x))[0].lang, StringType())

def remove_stopwords_spark(tokenized, in_col, out_col = None):
    if out_column is not None:
        remover = StopWordsRemover(inputCol = in_column, outputCol = out_column)
    else:
        remover = StopWordsRemover(inputCol = in_column)
    stopwords_removed = remover.transform(tokenized)
    return stopwords_removed

def detect_language_spark(df, in_col, out_col = None):
    if out_col is not None:
        df = df.withColumn(out_col, detect_language_udf(in_col))
    else:
        df = df.withColumn(in_col, detect_language_udf(in_col))
    return df
