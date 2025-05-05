import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from pyspark.sql.functions import col, sum, when, mean, udf, avg, variance, count, to_timestamp
from pyspark.ml.feature import OneHotEncoder, StringIndexer, MinMaxScaler
import findspark
from pyspark.sql import SparkSession

from api.utils import extract_time_features


def split_data(df): #create balanced test set
   # Initial random split
   train, test = df.randomSplit([0.8, 0.2], seed=42)

   # Balance the test set
   fraud_df_test = test.filter(col("TX_FRAUD") == 1)
   non_fraud_df_test = test.filter(col("TX_FRAUD") == 0)

   fraud_count_test = fraud_df_test.count()

   non_fraud_test_balanced = non_fraud_df_test \
      .sample(withReplacement=False, fraction=1.0, seed=42) \
      .limit(fraud_count_test)

   balanced_test = fraud_df_test.union(non_fraud_test_balanced)

   test_ids = balanced_test.select("TRANSACTION_ID")

   # Remove test samples from train to prevent leakage
   train = train.join(test_ids, on="TRANSACTION_ID", how="left_anti")

   return train, balanced_test


def smote(train_pd, target_column='TX_FRAUD'):
   X = train_pd.drop(columns=[target_column])
   y = train_pd[target_column]

   smo = SMOTE(random_state=42)
   X_resampled, y_resampled = smo.fit_resample(X, y)

   resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
   resampled_data[target_column] = y_resampled

   return resampled_data


def save_to_csv(df, filepath):

   df.to_csv(filepath, index=False)

def main():
   #spark
   findspark.init()
   spark = SparkSession.builder.master("local[*]").getOrCreate()
   spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
   print(spark)

   #data
   df = spark.read.csv('data/training_data.csv',inferSchema=True,header='true')
   df.printSchema()
   fraud_rate = df.groupBy("TX_FRAUD").count()
   fraud_rate.show()
   # extract time features
   df_time_extracted = extract_time_features(df)

   #balanced and fit data
   train,test = split_data(df_time_extracted)

   test.groupBy("TX_FRAUD").count().show()

   train.groupBy("TX_FRAUD").count().show()
   train_pd = train.toPandas()

   save_to_csv(smote(train_pd),'data/balanced_training_set.csv')
   save_to_csv(test.toPandas(),'data/test_set.csv')

   spark.stop()
if __name__ == '__main__':
   main()
