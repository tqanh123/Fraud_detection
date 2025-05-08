from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from xgboost.spark import SparkXGBClassifier
import numpy as np

class XGBoostClassifierWrapper:
    def __init__(self,
                 label_col="TX_FRAUD",
                 features_col="features",
                 num_workers=2,
                 missing=np.nan,
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
                 base_score=None,
                 booster=None,
                 callbacks=None,
                 colsample_bylevel=None,
                 colsample_bynode=None,
                 colsample_bytree=None,
                 device="cpu",
                 early_stopping_rounds=None,
                 eval_metric=None,
                 feature_names=None,
                 feature_types=None,
                 feature_weights=None,
                 force_repartition=False,
                 gamma=None,
                 grow_policy=None,
                 importance_type=None,
                 interaction_constraints=None,
                 iteration_range=None,
                 max_bin=None,
                 max_cat_threshold=None,
                 max_cat_to_onehot=None,
                 max_delta_step=None,
                 max_leaves=None,
                 min_child_weight=None,
                 monotone_constraints=None,
                 multi_strategy=None,
                 num_parallel_tree=None,
                 objective=None,
                 random_state=None,
                 reg_lambda=None,
                 repartition_random_shuffle=False,
                 sampling_method=None,
                 scale_pos_weight=None,
                 subsample=None,
                 tree_method=None,
                 validate_parameters=None,
                 verbosity=None,
                 xgb_model=None):

        self.label_col = label_col
        self.features_col = features_col
        self.num_workers = num_workers
        self.missing = missing
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.reg_alpha = reg_alpha
        self.verbose = verbose
        self.enable_sparse_data_optim = enable_sparse_data_optim
        self.use_gpu = use_gpu
        self.prediction_col = prediction_col
        self.probability_col = probability_col
        self.raw_prediction_col = raw_prediction_col
        self.base_score = base_score
        self.booster = booster
        self.callbacks = callbacks
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.colsample_bytree = colsample_bytree
        self.device = device
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.feature_weights = feature_weights
        self.force_repartition = force_repartition
        self.gamma = gamma
        self.grow_policy = grow_policy
        self.importance_type = importance_type
        self.interaction_constraints = interaction_constraints
        self.iteration_range = iteration_range
        self.max_bin = max_bin
        self.max_cat_threshold = max_cat_threshold
        self.max_cat_to_onehot = max_cat_to_onehot
        self.max_delta_step = max_delta_step
        self.max_leaves = max_leaves
        self.min_child_weight = min_child_weight
        self.monotone_constraints = monotone_constraints
        self.multi_strategy = multi_strategy
        self.num_parallel_tree = num_parallel_tree
        self.objective = objective
        self.random_state = random_state
        self.reg_lambda = reg_lambda
        self.repartition_random_shuffle = repartition_random_shuffle
        self.sampling_method = sampling_method
        self.scale_pos_weight = scale_pos_weight
        self.subsample = subsample
        self.tree_method = tree_method
        self.validate_parameters = validate_parameters
        self.verbosity = verbosity
        self.xgb_model = xgb_model

        self.model = self._create_model()

    def _create_model(self):
        return SparkXGBClassifier(
            label_col=self.label_col,
            features_col=self.features_col,
            num_workers=self.num_workers,
            missing=self.missing,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            reg_alpha=self.reg_alpha,
            verbose=self.verbose,
            enable_sparse_data_optim=self.enable_sparse_data_optim,
            use_gpu=self.use_gpu,
            prediction_col=self.prediction_col,
            probability_col=self.probability_col,
            raw_prediction_col=self.raw_prediction_col,
            base_score=self.base_score,
            booster=self.booster,
            callbacks=self.callbacks,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            colsample_bytree=self.colsample_bytree,
            device=self.device,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric=self.eval_metric,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            feature_weights=self.feature_weights,
            force_repartition=self.force_repartition,
            gamma=self.gamma,
            grow_policy=self.grow_policy,
            importance_type=self.importance_type,
            interaction_constraints=self.interaction_constraints,
            iteration_range=self.iteration_range,
            max_bin=self.max_bin,
            max_cat_threshold=self.max_cat_threshold,
            max_cat_to_onehot=self.max_cat_to_onehot,
            max_delta_step=self.max_delta_step,
            max_leaves=self.max_leaves,
            min_child_weight=self.min_child_weight,
            monotone_constraints=self.monotone_constraints,
            multi_strategy=self.multi_strategy,
            num_parallel_tree=self.num_parallel_tree,
            objective=self.objective,
            random_state=self.random_state,
            reg_lambda=self.reg_lambda,
            repartition_random_shuffle=self.repartition_random_shuffle,
            sampling_method=self.sampling_method,
            scale_pos_weight=self.scale_pos_weight,
            subsample=self.subsample,
            tree_method=self.tree_method,
            validate_parameters=self.validate_parameters,
            verbosity=self.verbosity,
            xgb_model=self.xgb_model
        )

    def get_model(self):
        return self.model

    def fit(self, df_train, df_test, numerical_cols):
        # VectorAssembler + StandardScaler
        assembler = VectorAssembler(inputCols=numerical_cols, outputCol="num_features")
        scaler = StandardScaler(inputCol='num_features', outputCol='features')

        # Pipeline
        pipeline = Pipeline(stages=[assembler, scaler])
        pipeline_model = pipeline.fit(df_train)

        df_train_transformed = pipeline_model.transform(df_train)
        df_test_transformed = pipeline_model.transform(df_test)
        df_train_transformed = df_train_transformed.select('features', 'TX_FRAUD')
        df_test_transformed = df_test_transformed.select('features', 'TX_FRAUD')

        xgb_model = self.model.fit(df_train_transformed)
        return xgb_model, df_train_transformed, df_test_transformed