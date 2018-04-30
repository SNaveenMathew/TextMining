from pyspark.ml.feature import StopWordsRemover
from util import detect_language, spell_correct_tokens
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType

def flatten_list(lis):
    out = []
    for l in lis:
        out = out + l
    return out

detect_language_udf = udf(lambda x: detect_language(" ".join(x))[0].lang, StringType())
spell_correct_tokens_udf = udf(lambda x: [spell_correct_tokens(y) for y in x], ArrayType(elementType = ArrayType(elementType = StringType())))
flatten_list_of_tokens_udf = udf(lambda x: flatten_list(x), ArrayType(elementType = StringType()))

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

def spell_correct_tokens_spark(df, in_col, out_col = None):
    if out_col is not None:
        df = df.withColumn(out_col, spell_correct_tokens_udf(in_col))
    else:
        df = df.withColumn(in_col, spell_correct_tokens_udf(in_col))
    return df

def flatten_list_of_tokens(df, in_col, out_col = None):
    if out_col is not None:
        df = df.withColumn(out_col, flatten_list_of_tokens_udf(in_col))
    else:
        df = df.withColumn(in_col, flatten_list_of_tokens_udf(in_col))
    return df
