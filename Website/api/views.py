import io
import os
import pickle
import pandas as pd
from django.shortcuts import render,redirect,get_object_or_404
from django.http.response import HttpResponseRedirect,HttpResponse,JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from pyspark.sql import SparkSession

from .models import DataFileUpload
from .utils import extract_time_features, load_model

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, date_format, count, lit

import logging 
import os

logger = logging.getLogger(__name__)
# $env:SPARK_HOME = "D:/App/spark-3.5.5-bin-hadoop3"
# $env:HADOOP_HOME = "D:/App/hadoop"
# $env:JAVA_HOME = "C:/Program Files/Java/jdk-24"
# $env:PYSPARK_PYTHON = "..\\venv\\Scripts\\python.exe"
# $env:PYSPARK_DRIVER_PYTHON = "..\\venv\\Scripts\\python.exe"

os.environ["SPARK_HOME"] = "D:/App/spark-3.5.5-bin-hadoop3"
os.environ["HADOOP_HOME"] = "D:/App/hadoop"
os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk-1.8"
os.environ["PYSPARK_PYTHON"] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "venv", "Scripts", "python.exe")
os.environ["PYSPARK_DRIVER_PYTHON"] = os.environ["PYSPARK_PYTHON"]

def base(request):
    return render(request,'api/landing_page.html')
    
def upload_credit_data(request):
    return render(request,'api/upload_credit_data.html')

def prediction_button(request, id):
    # Set environment variables
    os.environ["PYSPARK_PYTHON"] = "..\\venv\\Scripts\\python.exe"
    os.environ["PYSPARK_DRIVER_PYTHON"] = "..\\venv\\Scripts\\python.exe"

    try:
        # Load the data file
        obj = DataFileUpload.objects.get(id=id)
        spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
        df = spark.read.csv(obj.actual_file.path, header=True, inferSchema=True)

        # Extract time features (if needed for the model)
        df = extract_time_features(df)

        # Load the trained model
        trained_model = load_model('ModelTraining/LogisticRegressionModel')

        # Make predictions
        predictions = trained_model.transform(df)
        print("Model loaded successfully:", trained_model)

        # Convert predictions to Pandas DataFrame
        pandas_df = predictions.toPandas()

        # Prepare data for the bar chart
        # Extract fraud probabilities (x = not fraud, y = fraud)
        if "probability" in predictions.columns:
            pandas_df['not_fraud'] = pandas_df['probability'].apply(lambda prob: prob[0] * 100)  # Convert to percentage
            pandas_df['fraud'] = pandas_df['probability'].apply(lambda prob: prob[1] * 100)  # Convert to percentage

            # Group by CUSTOMER_ID and calculate average probabilities
            bar_data = pandas_df.groupby("CUSTOMER_ID")[["not_fraud", "fraud"]].mean().reset_index()

            # Prepare data for the bar chart
            bar_chart = {
                "categories": bar_data["CUSTOMER_ID"].tolist(),
                "series": [
                    {"name": "Not Fraud", "data": bar_data["not_fraud"].tolist()},
                    {"name": "Fraud", "data": bar_data["fraud"].tolist()}
                ]
            }
        else:
            bar_chart = {"categories": [], "series": []}

        print("Bar chart data:", bar_chart)

        # Drop unnecessary columns
        columns_to_drop = ['features', 'scaled_features', 'rawPrediction', 'probability', 'not_fraud', 'fraud']
        pandas_df = pandas_df.drop(columns=columns_to_drop, errors='ignore')


        # Pass data to the template
        return render(request, 'api/fraud_detection.html', {
            'id': id,
            'table_data': pandas_df.to_dict(orient="records"),
            'columns': pandas_df.columns.tolist(),
            'bar_chart': bar_chart
        })

    except FileNotFoundError as e:
        return HttpResponse(f"Error: {str(e)}", status=404)
    except RuntimeError as e:
        return HttpResponse(f"Error: {str(e)}", status=500)
    except Exception as e:
        return HttpResponse(f"An unexpected error occurred: {str(e)}", status=500)

# def prediction_button(request, id):
#     # Set environment variables
#     os.environ["PYSPARK_PYTHON"] = "..\\venv\\Scripts\\python.exe"
#     os.environ["PYSPARK_DRIVER_PYTHON"] = "..\\venv\\Scripts\\python.exe"

#     try:
#         # Load the data file
#         obj = DataFileUpload.objects.get(id=id)
#         spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
#         df = spark.read.csv(obj.actual_file.path, header=True, inferSchema=True)

#         # Extract time features (if needed for the model)
#         df = extract_time_features(df)

#         # Load the trained model
#         trained_model = load_model('ModelTraining/LogisticRegressionModel')

#         # Make predictions
#         predictions = trained_model.transform(df)
#         print("Model loaded successfully:", trained_model)

#         # Convert predictions to Pandas DataFrame
#         pandas_df = predictions.toPandas()

#         # Drop unnecessary columns
#         columns_to_drop = ['features', 'scaled_features', 'rawPrediction', 'probability']
#         pandas_df = pandas_df.drop(columns=columns_to_drop, errors='ignore')

#         # Prepare data for the bar chart
#         if "probability" in predictions.columns:
#             # Extract fraud probability (y value from [x, y])
#             pandas_df['fraud_probability'] = predictions.select("probability").rdd.map(lambda row: row.probability[1]).collect()

#             # Group by CUSTOMER_ID and calculate average fraud probability
#             bar_data = pandas_df.groupby("CUSTOMER_ID")['fraud_probability'].mean().reset_index()

#             bar_chart = {
#                 "categories": bar_data["CUSTOMER_ID"].tolist(),
#                 "series": [{"name": "Fraud Probability", "data": bar_data["fraud_probability"].tolist()}]
#             }
#         else:
#             bar_chart = {"categories": [], "series": []}

#         # Pass data to the template
#         return render(request, 'api/fraud_detection.html', {
#             'id': id,
#             'table_data': pandas_df.to_dict(orient="records"),
#             'columns': pandas_df.columns.tolist(),
#             'bar_chart': bar_chart
#         })

#     except FileNotFoundError as e:
#         return HttpResponse(f"Error: {str(e)}", status=404)
#     except RuntimeError as e:
#         return HttpResponse(f"Error: {str(e)}", status=500)
#     except Exception as e:
#         return HttpResponse(f"An unexpected error occurred: {str(e)}", status=500)


def reports(request):
    all_data_files_objs=DataFileUpload.objects.all()
    return render(request,'api/reports.html',{'all_files':all_data_files_objs})
    
def delete_data(request,id):
    obj=DataFileUpload.objects.get(id=id)
    obj.delete()
    messages.success(request, "File Deleted succesfully",extra_tags = 'alert alert-success alert-dismissible show')
    return HttpResponseRedirect('/reports')
def upload_data(request):
    if request.method == 'POST':
            data_file_name  = request.POST.get('data_file_name')
            try:
                actual_file = request.FILES['actual_file_name']
                description  = request.POST.get('description')

                DataFileUpload.objects.create(
                        file_name=data_file_name,
                        actual_file=actual_file,
                        description=description,
                    )
                
                
                messages.success(request, "File Uploaded succesfully",extra_tags = 'alert alert-success alert-dismissible show')
                return HttpResponseRedirect('/reports')
                
            except:
                messages.warning(request, "Invalid/wrong format. Please upload File.")
                return redirect('/upload_credit_data')
            
def view_data(request,id):
    obj = DataFileUpload.objects.get(id=id)
    df = pd.read_csv(obj.actual_file.path)
    columns = df.columns.tolist()
    return render(request,'api/view_data.html', {'id': id, 'columns': columns})

def retrieve_data_by_id(request, id):
    try:
        os.environ["PYSPARK_PYTHON"] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "venv", "Scripts", "python.exe")
        os.environ["PYSPARK_DRIVER_PYTHON"] = os.environ["PYSPARK_PYTHON"]
        
        logger.info("Checking Spark session status...")
        if SparkSession._instantiatedSession is None:
            logger.info("Spark session has NOT been initialized.")
        else:
            logger.info("Spark session is already initialized.")
            
        # Thêm thông tin về đường dẫn
        logger.info(f"SPARK_HOME: {os.environ.get('SPARK_HOME')}")
        logger.info(f"HADOOP_HOME: {os.environ.get('HADOOP_HOME')}")
        logger.info(f"JAVA_HOME: {os.environ.get('JAVA_HOME')}")
        logger.info(f"PYSPARK_PYTHON: {os.environ.get('PYSPARK_PYTHON')}")


        # Khởi tạo Spark session
        spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
        print("Spark session initialized successfully.")

        # Load the data
        obj = DataFileUpload.objects.get(id=id)
        df = spark.read.csv(obj.actual_file.path, header=True, inferSchema=True)

        # Check if '_c0' exists in the schema and drop it
        if '_c0' in df.columns:
            df = df.drop('_c0')


        # Receive parameters from the frontend
        customer_id = request.GET.get('customer_id')
        tx_month = request.GET.get('tx_month')

        # Convert TX_DATETIME to date and extract MM-YYYY
        df = df.withColumn("TX_DATETIME", to_date(col("TX_DATETIME"), "yyyy-MM-dd HH:mm:ss"))
        df = df.withColumn("TX_MONTH", date_format(col("TX_DATETIME"), "MM-yyyy"))

        # Filter by customer_id and tx_month
        if customer_id:
            df = df.filter(col("CUSTOMER_ID") == int(customer_id))
        if tx_month:
            df = df.filter(col("TX_MONTH") == tx_month)

        # Convert to Pandas for DataTables
        pandas_df = df.toPandas()

        # Prepare data for pie chart
        pie_data = pandas_df['TX_FRAUD'].value_counts(normalize=True) * 100
        pie_chart = {
            "labels": pie_data.index.tolist(),
            "values": pie_data.values.tolist()
        }
        print("Pie chart data: ", pie_chart)


        # Paginate the data for the table
        draw = int(request.GET.get('draw', 1))
        start = int(request.GET.get('start', 0))
        length = int(request.GET.get('length', 10))
        paginated_df = pandas_df.iloc[start:start + length].reset_index()
        paginated_df['index'] = paginated_df['index'] + 1 + start

        # Convert the paginated data to a list of lists
        data = paginated_df.values.tolist()

        # Return JSON response
        return JsonResponse({
            'draw': draw,
            'recordsTotal': len(pandas_df),
            'recordsFiltered': len(pandas_df),
            'data': data,
            'pie_chart': pie_chart
        })
    except Exception as e:
        logger.error(f"Error in retrieve_data_by_id: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)
    
    
def userLogout(request):
    try:
      del request.session['username']
    except:
      pass
    logout(request)
    return HttpResponseRedirect('/') 
    
def predict_fraud(request):
    if request.method == 'POST' and request.FILES.get('datafile'):

        uploaded_file = request.FILES['datafile']

        spark = SparkSession.builder.getOrCreate()

        test_df = spark.read.csv(uploaded_file.temporary_file_path(), header=True, inferSchema=True)

        test_df = extract_time_features(test_df)

        trained_model = load_model('ModelTraining/LogisticRegressionModel')

        predictions = trained_model.transform(test_df)
        #proba...
def login2(request):
    data = {}
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        print(user)
        if user:
            print("DEBUG — user.pk is:", user.pk)
            login(request, user)

            return HttpResponseRedirect('/')
        
        else:    
            data['error'] = "Username or Password is incorrect"
            res = render(request, 'api/login.html', data)
            return res
    else:
        return render(request, 'api/login.html', data)

def account_details(request):
    return render(request,'api/account_details.html')
def change_password(request):
    return render(request,'api/change_password.html')

def about(request):
    return render(request,'api/about.html')

def dashboard(request):
    return render(request,'api/dashboard.html')
