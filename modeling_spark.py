from pyspark.ml.feature import Word2Vec as w2v
from pickle import dump

def run_word2vec_model_pyspark(documentDF, in_col, vec_size = 3, in_type = "tokens", out_col = None):
    if in_type == "tokens": 
        if out_col is not None:
            model = w2v(vectorSize = vec_size, minCount = 0, inputCol = in_col, outputCol = out_col)
        else:
            model = w2v(vectorSize = vec_size, minCount = 0, inputCol = in_col)
        model = model.fit(documentDF)
        model.write().overwrite().save('w2v_spark.pkl')
        # Model save is working, but model load is not!
        result = model.transform(documentDF)
    
    # Implement this later if required
    # else:
    #     model = w2v(vectorSize = 3, minCount = 0, inputCol)
    return model, result
