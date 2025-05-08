from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

class RandomForestClassifierWrapper:
    def __init__(self,
                 labelCol="TX_FRAUD",
                 featuresCol="features",
                 numTrees=25,
                 maxDepth=5,
                 maxBins=32,
                 impurity="gini",
                 featureSubsetStrategy="auto",
                 subsamplingRate=1.0,
                 seed=-1463286261669475607,
                 maxMemoryInMB=256,
                 minInfoGain=0.0,
                 minInstancesPerNode=1,
                 minWeightFractionPerNode=0.0,
                 bootstrap=True,
                 cacheNodeIds=False,
                 checkpointInterval=10,
                 predictionCol="prediction",
                 probabilityCol="probability",
                 rawPredictionCol="rawPrediction",
                 leafCol=""):
        self.labelCol = labelCol
        self.featuresCol = featuresCol
        self.numTrees = numTrees
        self.maxDepth = maxDepth
        self.maxBins = maxBins
        self.impurity = impurity
        self.featureSubsetStrategy = featureSubsetStrategy
        self.subsamplingRate = subsamplingRate
        self.seed = seed
        self.maxMemoryInMB = maxMemoryInMB
        self.minInfoGain = minInfoGain
        self.minInstancesPerNode = minInstancesPerNode
        self.minWeightFractionPerNode = minWeightFractionPerNode
        self.bootstrap = bootstrap
        self.cacheNodeIds = cacheNodeIds
        self.checkpointInterval = checkpointInterval
        self.predictionCol = predictionCol
        self.probabilityCol = probabilityCol
        self.rawPredictionCol = rawPredictionCol
        self.leafCol = leafCol
        self.model = self._create_model()

    def _create_model(self):
        return RandomForestClassifier(
            labelCol=self.labelCol,
            featuresCol=self.featuresCol,
            numTrees=self.numTrees,
            maxDepth=self.maxDepth,
            maxBins=self.maxBins,
            impurity=self.impurity,
            featureSubsetStrategy=self.featureSubsetStrategy,
            subsamplingRate=self.subsamplingRate,
            seed=self.seed,
            maxMemoryInMB=self.maxMemoryInMB,
            minInfoGain=self.minInfoGain,
            minInstancesPerNode=self.minInstancesPerNode,
            minWeightFractionPerNode=self.minWeightFractionPerNode,
            bootstrap=self.bootstrap,
            cacheNodeIds=self.cacheNodeIds,
            checkpointInterval=self.checkpointInterval,
            predictionCol=self.predictionCol,
            probabilityCol=self.probabilityCol,
            rawPredictionCol=self.rawPredictionCol,
            leafCol=self.leafCol
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

        rf_model = self.model.fit(df_train_transformed)
        return rf_model, df_train_transformed, df_test_transformed