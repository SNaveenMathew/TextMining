from pyspark.ml.feature import StopWordsRemover, Normalizer
from util import detect_language, spell_correct_tokens
from pyspark.sql.functions import udf, monotonically_increasing_id
from pyspark.sql.types import StringType, ArrayType
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pandas import DataFrame

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

def get_semantic_similarity_spark(model):
    df = model.getVectors()
    normalizer = Normalizer(inputCol="vector", outputCol="norm")
    data = normalizer.transform(df)
    data = data.withColumn("ID", monotonically_increasing_id())
    mat = IndexedRowMatrix(
    data.select("ID", "norm")\
    .rdd.map(lambda row: IndexedRow(row.ID, row.norm.toArray()))).toBlockMatrix()
    sim1 = mat.multiply(mat.transpose())
    sim1 = DataFrame(sim1.toLocalMatrix().toArray())
    data_list = data.select('word').collect()
    sim1.columns = sim1.index = [str(i['word']) for i in data_list]
    return sim1
