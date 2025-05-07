import findspark
import pandas as pd
from matplotlib import pyplot as plt
from pyspark.sql.functions import col, when, sum as _sum
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, MinMaxScaler, RobustScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
import seaborn as sns
from LogisticRegression import LogisticRegressionPipeline
def main():

    findspark.init()

    spark = SparkSession.builder \
        .appName("ModelTrain") \
        .config("spark.hadoop.io.nativeio.enabled", "false") \
        .getOrCreate()

    spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
    print(spark)

    train_set = spark.read.csv('data/balanced_training_set.csv', inferSchema=True, header='true')
    test_set = spark.read.csv('data/test_set.csv', inferSchema=True, header='true')
    #train model pipeline
    lr_pipe = LogisticRegressionPipeline(
        target_col='TX_FRAUD',
        max_iter=10,
        reg_param=0.01,
        elastic_net=0.0,
    )

    # Fit the model
    fitted_model = lr_pipe.fit(train_set)

    # Make predictions
    predictions = lr_pipe.predict(test_set)
    predictions.show()
    fitted_model.save('LogisticRegressionModel')

if __name__ == '__main__':
    main()
