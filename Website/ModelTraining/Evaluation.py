import logging

import findspark
from pyspark.sql import SparkSession
import os
import sys

# from ModelTraining.SupportVectorMachine import SVMPipeline
# from Website.api.utils import metrics_calculator,model_evaluation,svm_evaluation
# from LogisticRegression import LogisticRegressionPipeline
from ExtremeGradientBoost import XGBoostClassifierWrapper
from RandomForest import RandomForestClassifierWrapper
import pyspark
findspark.init()

# Set up logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# Initialize Spark session

spark = SparkSession.builder \
    .appName("FUCKYOUXGBOOST") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

train_set = spark.read.csv('Website/ModelTraining/data/balanced_training_set.csv', inferSchema=True, header='true')
test_set = spark.read.csv('Website/ModelTraining/data/test_set.csv', inferSchema=True, header='true')

# svm_model = SVMPipeline(
#     label_col="TX_FRAUD",
#     features_col="features",
#     max_iter=50,
#     reg_param=0.1,
#     tol=1e-6,
#     fit_intercept=True,
#     standardization=True,
#     aggregation_depth=2,
#     max_block_size=0.0
# )
# print("Evaluation for Support Vector Machine:\n")
# svm_model.fit(train_set)
# svm_predictions = svm_model.predict(test_set)
# svm_evaluation(svm_predictions)
#
# lr_model = LogisticRegressionPipeline(
#     target_col='TX_FRAUD',
#     max_iter=20,
#     reg_param=0.01,
#     elastic_net=0.0
# )
# print("Evaluation for Logistic Regression:\n")
# lr_model.fit(train_set)
# lr_predictions = lr_model.predict(test_set)
# model_evaluation(lr_model,lr_predictions,"Logistic Regression")

train_set = train_set.drop('TX_FRAUD_SCENARIO')
test_set = test_set.drop('TX_FRAUD_SCENARIO')
numerical_cols = ['TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS', 'hour', 'day_of_week', 'month', 'minute', 'second']

# print("Evaluation for Random Forest:\n")
# rf_wrapper = RandomForestClassifierWrapper()
# rf_model, df_train_transformed, df_test_transformed = rf_wrapper.fit(train_set, test_set, numerical_cols)
# rf_wrapper.savepip('Website/ModelTraining/RandomForestModel')
#
# rf_predictions = rf_model.transform(df_test_transformed)
# model_evaluation(rf_model,rf_predictions,"Random Forest")

print("Evaluation for Extreme Gradient Boost Tree:\n")
xgb_wrapper = XGBoostClassifierWrapper()
xgb_model, df_train_transformed, df_test_transformed = xgb_wrapper.fit(train_set, test_set, numerical_cols)
xgb_wrapper.savepip('Website/ModelTraining/XGBoostModel')
loaded_pipeline_model, trained_rf_model = XGBoostClassifierWrapper.load_model_pip('Website/ModelTraining/XGBoostModel')
df = loaded_pipeline_model.transform(test_set)
# xgb_predictions = xgb_model.transform(df_test_transformed)
# model_evaluation(xgb_model,xgb_predictions,"XGBoost")

# metadata_dir = os.path.join('Website/ModelTraining/XGBoostModel/xgb_model', 'metadata')
# print(f"Listing contents of metadata directory: {metadata_dir}")
#
# metadata_files = os.listdir(metadata_dir)
# print("Metadata files:", metadata_files)

# part_file_path = os.path.join('Website/ModelTraining/XGBoostModel/xgb_model/metadata', 'part-00000-0f5ad03d-9372-4b5c-9ba5-7dd61e5006b2-c000.txt')
#
# with open(part_file_path, 'r') as f:
#     part_file_content = f.read()
#
# print(f"Part file content:\n{part_file_content}")


