import logging

import findspark
from pyspark.sql import SparkSession
import os
import sys

from ModelTraining.SupportVectorMachine import SVMPipeline
from Website.api.utils import metrics_calculator,model_evaluation,svm_evaluation
from LogisticRegression import LogisticRegressionPipeline
from ExtremeGradientBoost import XGBPipeline
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

train_set = spark.read.csv('data/balanced_training_set.csv', inferSchema=True, header='true')
test_set = spark.read.csv('data/test_set.csv', inferSchema=True, header='true')

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

print("Evaluation for Extreme Gradient Boost Tree:\n")
xgb_model = XGBPipeline(
    label_col='TX_FRAUD',
    num_boost_round=200,
    num_workers=2,
    max_depth=10,
    learning_rate=0.1,
    reg_alpha=0.1
)
xgb_model.fit(train_set)
xgb_predictions = xgb_model.predict(test_set)
model_evaluation(xgb_model,xgb_predictions,"XGBoost")





