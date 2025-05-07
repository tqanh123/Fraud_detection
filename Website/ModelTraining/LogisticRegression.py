from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, RobustScaler
from pyspark.ml.classification import LogisticRegression


class LogisticRegressionPipeline:
    def __init__(self, target_col, max_iter=20, reg_param=0.01, elastic_net=0.0):

        self.target_col = target_col
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.elastic_net = elastic_net
        self.model = None

    def build_pipeline(self, input_cols):

        assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
        scaler = RobustScaler(inputCol='features', outputCol='scaled_features')

        lr = LogisticRegression(
            featuresCol='scaled_features',
            labelCol=self.target_col,
            predictionCol='prediction',
            probabilityCol='probability',
            rawPredictionCol='rawPrediction',
            maxIter=self.max_iter,
            regParam=self.reg_param,
            elasticNetParam=self.elastic_net,
            fitIntercept=True,
            standardization=True,
            threshold=0.5,
            tol=1e-6,
            aggregationDepth=2,
        )

        pipeline = Pipeline(stages=[assembler, scaler, lr])
        return pipeline

    def fit(self, train_df):

        feature_cols = [col for col in train_df.columns if col != self.target_col]
        pipeline = self.build_pipeline(feature_cols)
        self.model = pipeline.fit(train_df)
        return self.model

    def predict(self, test_df):

        if self.model is None:
            raise ValueError("Model has not been trained. Call `fit()` first.")
        return self.model.transform(test_df)