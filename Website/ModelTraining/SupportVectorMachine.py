from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler, RobustScaler

class SVMPipeline:
    def __init__(self,
                 label_col="TX_FRAUD",
                 features_col="features",
                 scaled_col="scaled_features",
                 max_iter=50,
                 reg_param=0.1,
                 tol=1e-6,
                 fit_intercept=True,
                 standardization=True,
                 aggregation_depth=2,
                 max_block_size=0.0):
        self.label_col = label_col
        self.features_col = features_col
        self.scaled_col = scaled_col
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardization = standardization
        self.aggregation_depth = aggregation_depth
        self.max_block_size = max_block_size
        self.model = None

    def build_pipeline(self, input_cols):
        assembler = VectorAssembler(inputCols=input_cols, outputCol=self.features_col)
        scaler = RobustScaler(inputCol=self.features_col, outputCol=self.scaled_col)

        svc_params = {
            "featuresCol": self.scaled_col,
            "labelCol": self.label_col,
            "predictionCol": "prediction",
            "rawPredictionCol": "rawPrediction",
            "maxIter": self.max_iter,
            "regParam": self.reg_param,
            "tol": self.tol,
            "fitIntercept": self.fit_intercept,
            "standardization": self.standardization,
            "aggregationDepth": self.aggregation_depth
        }

        if self.max_block_size > 0:
            svc_params["maxBlockSizeInMB"] = self.max_block_size

        svc = LinearSVC(**svc_params)
        return Pipeline(stages=[assembler, scaler, svc])

    def fit(self, train_df):
        feature_cols = [col for col in train_df.columns if col != self.label_col]
        pipeline = self.build_pipeline(feature_cols)
        self.model = pipeline.fit(train_df)
        return self.model

    def predict(self, test_df):
        if self.model is None:
            raise ValueError("Model not trained. Call 'fit()' first.")
        return self.model.transform(test_df)
