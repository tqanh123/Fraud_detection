import os
from pyspark.ml import PipelineModel
from pyspark.sql.functions import hour, minute, second, dayofweek, month, year, to_timestamp, col


def extract_time_features(df):
    df = df.withColumn('TX_DATETIME', to_timestamp(col('TX_DATETIME'), 'yyyy-MM-dd HH:mm:ss'))
    df_time_extracted = df.withColumn('hour', hour('TX_DATETIME')) \
        .withColumn('minute', minute('TX_DATETIME')) \
        .withColumn('second', second('TX_DATETIME')) \
        .withColumn('day_of_week', dayofweek('TX_DATETIME') - 2) \
        .withColumn('month', month('TX_DATETIME')) \
        .withColumn('year', year('TX_DATETIME'))

    return df_time_extracted.drop("TX_DATETIME")

def load_model(model_path):
    if os.path.exists(model_path):
        try:
            model = PipelineModel.load(model_path)
            return model
        except Exception as e:
            return f"Error loading model: {str(e)}"
    else:
        return "Model path does not exist"