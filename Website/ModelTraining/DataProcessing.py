import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from pyspark.sql.functions import col, sum, when, mean, udf, avg, variance, count, to_timestamp
from pyspark.ml.feature import OneHotEncoder, StringIndexer, MinMaxScaler
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

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



def count_nulls(spark_df):
    """
    Returns a Spark DataFrame containing the count of null (all types) or empty string values (strings only) per column.
    """
    exprs = []
    for field in spark_df.schema:
        c = field.name
        if isinstance(field.dataType, StringType):
            cond = when(col(c).isNull() | (col(c) == ""), 1).otherwise(0)
        else:
            cond = when(col(c).isNull(), 1).otherwise(0)
        exprs.append(sum(cond).alias(c))
    return spark_df.select(exprs)

def count_duplicates(spark_df) -> int:

   total_rows = spark_df.count()
   distinct_rows = spark_df.distinct().count()
   return total_rows - distinct_rows

def detect_outliers(spark_df, column):

   Q1, Q3 = spark_df.approxQuantile(column, [0.25, 0.75], 0)
   IQR = Q3 - Q1
   outliers = spark_df.filter((col(column) < (Q1 - 1.5 * IQR)) | (col(column) > (Q3 + 1.5 * IQR)))
   num_outliers = outliers.count()
   if 'TX_FRAUD' in spark_df.columns:
      fraud_rate = outliers.agg(mean("TX_FRAUD")).collect()[0][0]
      return num_outliers, fraud_rate
   return num_outliers

def plot_boxplots(pd_df, numeric_columns, n_cols=3, figsize=(20, 5)):

    n_rows = -(-len(numeric_columns) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    axes = axes.flatten()
    for i, column in enumerate(numeric_columns):
        sns.boxplot(y=pd_df[column], ax=axes[i])
        axes[i].set_title(f"Boxplot for {column}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel(column)
    for j in range(len(numeric_columns), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def distribution_by_category(spark_df, category_col):

   return (
      spark_df.groupBy(category_col, "TX_FRAUD")
      .count()
      .withColumnRenamed("count", "Total")
   )


def plot_count_by_time(pd_df, time_col: str, fraud_col='TX_FRAUD'):

   prob_df = pd_df.groupby(time_col)[fraud_col].value_counts(normalize=True).rename('Probability').reset_index()
   sns.barplot(x=time_col, y="Probability", hue=fraud_col, data=prob_df)
   plt.title(f"Fraud Probability by {time_col}")
   plt.xlabel(time_col)
   plt.ylabel("Probability")
   plt.show()


def plot_fraud_rate_pie(fraud_rate_pd):

   fraud_rate_pd.set_index('TX_FRAUD', inplace=True)
   fraud_rate_pd['Percentage'].plot(kind='pie', autopct='%1.1f%%', figsize=(6, 6))
   plt.title("Fraud Rate Distribution")
   plt.ylabel('')
   plt.show()


def plot_histograms(pd_df):
   """
   Plots a histogram with KDE for the 'tx_amount' column in a pandas DataFrame using density (probability).
   """
   plt.figure(figsize=(10, 5))
   sns.histplot(pd_df['TX_AMOUNT'], kde=True, bins=30, stat="density")
   plt.title("Distribution of tx_amount")
   plt.xlabel("tx_amount")
   plt.ylabel("Density")
   plt.tight_layout()
   plt.show()

def main():
   #spark
   findspark.init()
   spark = SparkSession.builder.master("local[*]").getOrCreate()
   spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
   print(spark)

   #data
   df = spark.read.csv('data/training_data.csv',inferSchema=True,header='true')
   df = df.withColumn('TX_DATETIME', to_timestamp(col('TX_DATETIME'), 'yyyy-MM-dd HH:mm:ss'))
   df.printSchema()
   fraud_rate = df.groupBy("TX_FRAUD").count()
   fraud_rate.show()

   df = extract_time_features(df)
   df.printSchema()

   #EDA
   # 1. Count nulls
   nulls_df = count_nulls(df)
   nulls_df.show()

   # 2. Count duplicates
   dupe_count = count_duplicates(df)
   print(f"Duplicate rows: {dupe_count}")

   # 3. Boxplots for numeric columns
   num_cols = ['TX_AMOUNT','TX_TIME_SECONDS','TX_TIME_DAYS']
   plot_boxplots(df.toPandas(), num_cols)

   # 4. Outlier detection on TX_AMOUNT
   num_outliers, fraud_rate = detect_outliers(df, "TX_AMOUNT")
   print(f"TX_AMOUNT → outliers: {num_outliers}, fraud rate: {fraud_rate:.2%}")

   # 5. Fraud distribution by scenario
   dist_scenario = distribution_by_category(df, "TX_FRAUD_SCENARIO")
   dist_scenario.show()

   # 6. Time-based countplots (hour, day_of_week, etc.)
   plot_count_by_time(df.toPandas(), "hour")
   plot_count_by_time(df.toPandas(), "minute")
   plot_count_by_time(df.toPandas(), "second")
   plot_count_by_time(df.toPandas(), "day_of_week")
   plot_count_by_time(df.toPandas(), "month")
   plot_count_by_time(df.toPandas(), "year")

   # 7. Pie chart of overall fraud rate
   fraud_rate = df.groupBy("TX_FRAUD").count().withColumn("Percentage", (col("count") / df.count()) * 100)
   fraud_rate = fraud_rate.toPandas()
   plot_fraud_rate_pie(fraud_rate)

   # 8. Histograms for numeric distributions
   plot_histograms(df.toPandas())

   # #balanced and fit data
   # train,test = split_data(df_time_extracted)
   # #
   # test.groupBy("TX_FRAUD").count().show()
   # #
   # train.groupBy("TX_FRAUD").count().show()
   #
   # #drop TX_FRAUD_SCENARIO
   # train = train.drop('TX_FRAUD_SCENARIO')
   # test = test.drop('TX_FRAUD_SCENARIO')
   # train_pd = train.toPandas()
   #
   # save_to_csv(smote(train_pd),'data/balanced_training_set.csv')
   # save_to_csv(test.toPandas(),'data/test_set.csv')


   spark.stop()
if __name__ == '__main__':
   main()
