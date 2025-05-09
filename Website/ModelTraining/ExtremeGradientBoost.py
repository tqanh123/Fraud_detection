from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from xgboost.spark import SparkXGBClassifier, SparkXGBClassifierModel
import numpy as np
import os

class XGBoostClassifierWrapper:
    def __init__(self,
                 label_col="TX_FRAUD",
                 features_col="features",
                 num_workers=2,
                 missing=0,
                 learning_rate=0.1,
                 max_depth=10,
                 n_estimators=200,
                 reg_alpha=0.1,
                 verbose=True,
                 enable_sparse_data_optim=False,
                 use_gpu=False,
                 prediction_col="prediction",
                 probability_col="probability",
                 raw_prediction_col="rawPrediction",
                 device="cpu",
                 force_repartition=False,
                 repartition_random_shuffle=False,
                 xgb_model=None):

        self.label_col = label_col
        self.features_col = features_col
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.reg_alpha = reg_alpha
        self.verbose = verbose
        self.enable_sparse_data_optim = enable_sparse_data_optim
        self.use_gpu = use_gpu
        self.device = device
        self.prediction_col = prediction_col
        self.probability_col = probability_col
        self.raw_prediction_col = raw_prediction_col
        self.force_repartition = force_repartition
        self.missing = missing
        self.repartition_random_shuffle = repartition_random_shuffle
        self.xgb_model = xgb_model
        self.model = self._create_model()
        self.pipeline_model = None
        self.trained_model = None

    def _create_model(self):
        params = {
            "label_col": self.label_col,
            "features_col": self.features_col,
            "num_workers": self.num_workers,
            "missing": self.missing,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "reg_alpha": self.reg_alpha,
            "verbose": self.verbose,
            "enable_sparse_data_optim": self.enable_sparse_data_optim,
            "use_gpu": self.use_gpu,
            "prediction_col": self.prediction_col,
            "probability_col": self.probability_col,
            "raw_prediction_col": self.raw_prediction_col,
            "device": self.device,
            "force_repartition": self.force_repartition,
            "repartition_random_shuffle": self.repartition_random_shuffle,
            "xgb_model": self.xgb_model
        }

        # Remove keys with None values
        filtered_params = {k: v for k, v in params.items() if v is not None}

        return SparkXGBClassifier(**filtered_params)

    def get_model(self):
        return self.model

    def fit(self, df_train, df_test, numerical_cols):
    # VectorAssembler + StandardScaler
        assembler = VectorAssembler(inputCols=numerical_cols, outputCol="num_features")
        scaler = StandardScaler(inputCol='num_features', outputCol='features')

        # Pipeline
        pipeline = Pipeline(stages=[assembler, scaler])
        self.pipeline_model = pipeline.fit(df_train)

        df_train_transformed = self.pipeline_model.transform(df_train)
        df_test_transformed = self.pipeline_model.transform(df_test)
        df_train_transformed = df_train_transformed.select('features', 'TX_FRAUD')
        df_test_transformed = df_test_transformed.select('features', 'TX_FRAUD')

        xgb_model = self.model.fit(df_train_transformed)
        self.trained_model = xgb_model
        return xgb_model, df_train_transformed, df_test_transformed

    def savepip(self, path):
        pipeline_path = os.path.join(path, "pipeline")
        xgb_model_path = os.path.join(path, "xgb_model")

        # Create directories if they do not exist
        if not os.path.exists(pipeline_path):
            os.makedirs(pipeline_path)
            print(f"Created directory: {pipeline_path}")

        if not os.path.exists(xgb_model_path):
            os.makedirs(xgb_model_path)
            print(f"Created directory: {xgb_model_path}")

        # Save the pipeline model with overwrite option
        if self.pipeline_model:
            print(f"Saving pipeline model to {pipeline_path}")
            self.pipeline_model.write().overwrite().save(pipeline_path)  # Use overwrite() here
        else:
            raise ValueError("Pipeline model is not trained yet. Call `fit()` before saving.")

        # Save the random forest model with overwrite option
        if self.trained_model:
            print(f"Saving trained model to {xgb_model_path}")
            self.trained_model.write().overwrite().save(xgb_model_path)  # ✅ Save trained model
        else:
            raise ValueError("Trained model not found. Call `fit()` before saving.")

    @staticmethod
    def load_model_pip(path):
        try:
            pipeline_model = PipelineModel.load(os.path.join(path, "pipeline"))
            xgb_model = SparkXGBClassifierModel.load(os.path.join(path, "xgb_model"))
            print(f"Model loaded successfully from {path}")
            return pipeline_model, xgb_model
        except Exception as e:
            print(f"Error loading model from {path}: {str(e)}")
            raise TypeError(f"Loaded model is not valid. Got: {str(e)}")