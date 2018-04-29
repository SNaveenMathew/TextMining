from pos_tagging import run_treetagger_pos_tag_text
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

def run_treetagger_pos_tag_spark(df, in_col, out_col = None, get_lemma = False):
    run_treetagger_pos_tag_text_udf = udf(lambda x: [run_treetagger_pos_tag_text(y, get_lemma = get_lemma) for y in x], ArrayType(elementType = ArrayType(elementType = ArrayType(elementType = StringType()))))
    if out_col is not None:
        df = df.withColumn(out_col, run_treetagger_pos_tag_text_udf(in_col))
    else:
        df = df.withColumn(in_col, run_treetagger_pos_tag_text_udf(in_col))
    return df
