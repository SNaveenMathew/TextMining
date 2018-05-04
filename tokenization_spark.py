from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType
from nltk.tokenize import sent_tokenize
from tokenization import tokenize_treetagger

# def tokenize_spark(sentenceDataFrame, in_column, out_column = None):
#     if out_column is not None:
#         tokenizer = Tokenizer(inputCol = in_column, outputCol = out_column)
#     else:
#         tokenizer = Tokenizer(inputCol = in_column, outputCol = in_column)
#     tokenized = tokenizer.transform(sentenceDataFrame)
#     return tokenized

sent_tokenize_udf = udf(lambda x: [y for y in sent_tokenize(x) if y!=""], ArrayType(elementType = StringType()))

def tokenize_sentence_nltk_spark(df, in_col, out_col = None):
    df = df.withColumn(in_col, regexp_replace(str = in_col, pattern = "\n", replacement = ". "))
    df = df.withColumn(in_col, regexp_replace(str = in_col, pattern = "\xa0", replacement = " "))
    if out_col is not None:
        df = df.withColumn(out_col, sent_tokenize_udf(in_col))
    else:
        df = df.withColumn(in_col, sent_tokenize_udf(in_col))
    return df

def non_null_tokens(x, get_lemma):
    ret = [tokenize_treetagger(y, get_lemma = get_lemma) for y in x if y!= ""]
    ret = [token for token in ret if token[0]!=""]
    return ret

def tokenize_treetagger_spark(df, in_col, out_col = None, get_lemma = False):
    tokenize_treetagger_udf = udf(lambda x: non_null_tokens(x), ArrayType(elementType = StringType()))
    if out_col is not None:
        df = df.withColumn(out_col, tokenize_treetagger_udf(in_col))
    else:
        df = df.withColumn(in_col, tokenize_treetagger_udf(in_col))
    return df
