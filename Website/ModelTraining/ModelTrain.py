import findspark

from pyspark.sql import SparkSession

from LogisticRegression import LogisticRegressionPipeline
from ModelTraining.SupportVectorMachine import SVMPipeline


def main():

    findspark.init()
    import sys
    print(f"Python version: {sys.version}")
    import os

    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("FraudDetection") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.python.worker.memory", "512m") \
        .getOrCreate()

    print(spark)

    train_set = spark.read.csv('data/balanced_training_set.csv', inferSchema=True, header='true')
    test_set = spark.read.csv('data/test_set.csv', inferSchema=True, header='true')


    # #train model pipeline
    lr_model = LogisticRegressionPipeline(
        target_col='TX_FRAUD',
        max_iter=20,
        reg_param=0.01,
        elastic_net=0.0
    )
    # # Fit the model
    lr = lr_model.fit(train_set)
    #
    # Make predictions
    predictions = lr_model.predict(test_set)
    predictions.show()

    svm_model = SVMPipeline(
        label_col="TX_FRAUD",
        features_col="features",
        max_iter=50,
        reg_param=0.1,
        tol=1e-6,
        fit_intercept=True,
        standardization=True,
        aggregation_depth=2,
        max_block_size=0.0
    )
    svm = svm_model.fit(train_set)
    svm.save('SVM')

if __name__ == '__main__':
    main()
