import findspark
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, MinMaxScaler, RobustScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

def metrics(predictions):

    df = predictions.select(
        col("prediction").cast("double").alias("prediction"),
        col("TX_FRAUD").cast("double").alias("TX_FRAUD")
    )

    #TP, FP, FN for class 1.0 (fraud)
    tp = df.filter((col("prediction") == 1.0) & (col("TX_FRAUD") == 1.0)).count()
    fp = df.filter((col("prediction") == 1.0) & (col("TX_FRAUD") == 0.0)).count()
    fn = df.filter((col("prediction") == 0.0) & (col("TX_FRAUD") == 1.0)).count()

    #precision and recall for fraud (class 1.0)
    precision_fraud = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_fraud = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_fraud = 2 * (precision_fraud * recall_fraud) / (precision_fraud + recall_fraud) if (precision_fraud + recall_fraud) > 0 else 0.0


    # TP, FP, FN for class 0.0 (no-fraud)
    tp_no_fraud = df.filter((col("prediction") == 0.0) & (col("TX_FRAUD") == 0.0)).count()  # True negatives
    fp_no_fraud = df.filter((col("prediction") == 0.0) & (col("TX_FRAUD") == 1.0)).count()  # False negatives for 0
    fn_no_fraud = df.filter((col("prediction") == 1.0) & (col("TX_FRAUD") == 0.0)).count()  # False positives for 0

    #precision, recall, and F1-score for no-fraud (class 0.0)
    precision_no_fraud = tp_no_fraud / (tp_no_fraud + fp_no_fraud) if (tp_no_fraud + fp_no_fraud) > 0 else 0.0
    recall_no_fraud = tp_no_fraud / (tp_no_fraud + fn_no_fraud) if (tp_no_fraud + fn_no_fraud) > 0 else 0.0
    f1_no_fraud = 2 * (precision_no_fraud * recall_no_fraud) / (precision_no_fraud + recall_no_fraud) if (precision_no_fraud + recall_no_fraud) > 0 else 0.0

    # Specificity (True Negative Rate) for no-fraud (class 0.0)
    specificity_no_fraud = tp_no_fraud / (tp_no_fraud + fn_no_fraud) if (tp_no_fraud + fn_no_fraud) > 0 else 0.0

    return f1_fraud,precision_fraud,recall_fraud, f1_no_fraud,precision_no_fraud, recall_no_fraud,specificity_no_fraud

def main():

    findspark.init()

    spark = SparkSession.builder \
        .appName("ModelTrain") \
        .config("spark.hadoop.io.nativeio.enabled", "false") \
        .getOrCreate()

    spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
    print(spark)

    train_set = spark.read.csv('data/balanced_training_set.csv', inferSchema=True, header='true')

    #train model pipeline
    feature_columns = [col for col in train_set.columns if col != 'TX_FRAUD']

    assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

    scaler = RobustScaler(inputCol='features', outputCol='scaled_features')

    lr = LogisticRegression(
        featuresCol='scaled_features',
        labelCol='TX_FRAUD',
        predictionCol='prediction',
        probabilityCol='probability',
        rawPredictionCol='rawPrediction',
        maxIter=10,
        regParam=0.01,
        elasticNetParam=0.0,
        fitIntercept=True,
        standardization=True,
        threshold=0.5,
        tol=1e-6,
        aggregationDepth=2,
        family='auto',
        maxBlockSizeInMB=0.0
    )

    pipeline = Pipeline(stages=[assembler, scaler, lr])
    model = pipeline.fit(train_set)
    model.save('LogisticRegressionModel')

if __name__ == '__main__':
    main()
