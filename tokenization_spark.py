from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, ArrayType
from nltk.tokenize import sent_tokenize

# def tokenize_spark(sentenceDataFrame, in_column, out_column = None):
#     if out_column is not None:
#         tokenizer = Tokenizer(inputCol = in_column, outputCol = out_column)
#     else:
#         tokenizer = Tokenizer(inputCol = in_column, outputCol = in_column)
#     tokenized = tokenizer.transform(sentenceDataFrame)
#     return tokenized

sent_tokenize_udf = udf(lambda x: sent_tokenize(x), ArrayType(elementType = StringType()))

def tokenize_sentence_nltk_spark(df, in_col):
    df = df.withColumn(in_col, regexp_replace(str = in_col, pattern = "\n", replacement = ". "))
    df = df.withColumn(in_col, regexp_replace(str = in_col, pattern = "\xa0", replacement = " "))
    df = df.withColumn(in_col, sent_tokenize_udf(in_col))
    return df
