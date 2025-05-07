import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import hour, minute, second, dayofweek, month, year, to_timestamp, col
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report, confusion_matrix, roc_curve
import seaborn as sns

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

def metrics_calculator(y_true, y_pred_class, y_pred_prob, model_name):
    '''
    This function calculates all desired performance metrics for a given model.
    '''
    result = pd.DataFrame(data=[
        accuracy_score(y_true, y_pred_class),
        precision_score(y_true, y_pred_class, average='binary', zero_division=1),
        recall_score(y_true, y_pred_class, average='binary', zero_division=1),
        f1_score(y_true, y_pred_class, average='binary', zero_division=1),
        roc_auc_score(y_true, y_pred_prob)  # Use probabilities for AUC
    ],
    index=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC'],
    columns=[model_name])

    result = (result * 100).round(2).astype(str) + '%'
    return result

def model_evaluation(clf, test_predictions, model_name):

    # Extract true labels
    y_true = [row['TX_FRAUD'] for row in test_predictions.select("TX_FRAUD").collect()]
    y_pred_prob = [row['probability'][1] for row in test_predictions.select("probability").collect()]
    y_pred_class = [1 if prob >= 0.5 else 0 for prob in y_pred_prob]

    # Classification report
    print(f"\n\t  Classification report for test set for {model_name}:")
    print("-" * 55)
    print(classification_report(y_true, y_pred_class))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Fraud', 'Fraud'],
                yticklabels=['No Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve and AUC
    if y_pred_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        auc = roc_auc_score(y_true, y_pred_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc='lower right')
        plt.show()
    else:
        print("ROC Curve and AUC are not available for models without probability estimates.")

    # Performance metrics
    result = metrics_calculator(y_true, y_pred_class, y_pred_prob, model_name)
    print(result)

def run_prediction(spark_df):
    # Extract time features
    df = extract_time_features(spark_df)

    # Load model
    trained_model = load_model('ModelTraining/LogisticRegressionModel')

    # Predict
    predictions = trained_model.transform(df)

    # Convert to Pandas
    pd_df = predictions.toPandas()

    # Bar chart data
    if "probability" in predictions.columns:
        pd_df['not_fraud'] = pd_df['probability'].apply(lambda prob: prob[0] * 100)
        pd_df['fraud'] = pd_df['probability'].apply(lambda prob: prob[1] * 100)

        bar_data = pd_df.groupby("CUSTOMER_ID")[["not_fraud", "fraud"]].mean().reset_index()

        bar_chart = {
            "categories": bar_data["CUSTOMER_ID"].tolist(),
            "series": [
                {"name": "Not Fraud", "data": bar_data["not_fraud"].tolist()},
                {"name": "Fraud", "data": bar_data["fraud"].tolist()}
            ]
        }
    else:
        bar_chart = {"categories": [], "series": []}

    # Drop Spark-specific columns
    columns_to_drop = ['features', 'scaled_features', 'rawPrediction', 'probability', 'not_fraud', 'fraud']
    pd_df = pd_df.drop(columns=columns_to_drop, errors='ignore')

    return pd_df, bar_chart

def svm_evaluation(test_predictions):

    evaluator = MulticlassClassificationEvaluator(
        labelCol="TX_FRAUD",
        predictionCol="prediction"
    )
    # Compute evaluation metrics
    accuracy = evaluator.setMetricName("accuracy").evaluate(test_predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(test_predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(test_predictions)
    f1_score = evaluator.setMetricName("f1").evaluate(test_predictions)

    # Print the evaluation metrics
    print("\nEvaluation Metrics for LinearSVC Model:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")