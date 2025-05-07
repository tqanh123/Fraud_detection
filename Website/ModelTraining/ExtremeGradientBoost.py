import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, RobustScaler
from xgboost.spark import SparkXGBClassifier
import pandas as pd

class XGBPipeline:
    def __init__(self,
                 label_col="TX_FRAUD",
                 num_boost_round=200,
                 num_workers=2,
                 missing=np.nan,
                 max_depth=10,
                 learning_rate=0.1,
                 reg_alpha=0.1):
        self.label_col = label_col
        self.num_boost_round = num_boost_round
        self.num_workers = num_workers
        self.missing = missing
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.model = None
    def build_pipeline(self, input_cols):
        assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
        scaler = RobustScaler(inputCol='features', outputCol='scaled_features')

        xgb = SparkXGBClassifier(
            label_col=self.label_col,
            features_col='scaled_features',
            num_workers=self.num_workers,
            missing=self.missing,
            num_boost_round=self.num_boost_round,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_alpha=self.reg_alpha,
            prediction_col="prediction",
            probability_col="probability",
        )
        pipeline = Pipeline(stages=[assembler, scaler, xgb])
        return pipeline

    def fit(self, train_df):
        feature_cols = [col for col in train_df.columns if col != self.label_col]
        pipeline = self.build_pipeline(feature_cols)
        self.model = pipeline.fit(train_df)
        return self.model

    def predict(self, test_data):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call 'fit' first.")
        predictions = self.model.transform(test_data)
        return predictions