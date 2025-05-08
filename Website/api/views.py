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
from .utils import extract_time_features, load_model, run_prediction

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, date_format, count, lit


def base(request):
    return render(request,'api/landing_page.html')
    
def upload_credit_data(request):
    return render(request,'api/upload_credit_data.html')
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
            
def view_data(request, id):
    obj = DataFileUpload.objects.get(id=id)
    df = pd.read_csv(obj.actual_file.path)
    columns = df.columns.tolist()
    data = df.to_dict(orient='records')  # Convert rows to list of dicts
    return render(request, 'api/view_data.html', {
        'id': id,
        'columns': columns,
        'data': data
    })

def retrieve_data_by_id(request, id):

    # Initialize PySpark session
    spark = SparkSession.builder.appName("FraudDetection").getOrCreate()

    # Load the data
    obj = DataFileUpload.objects.get(id=id)
    df = spark.read.csv(obj.actual_file.path, header=True, inferSchema=True)

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

    #Data for table
    pandas_df.insert(0, "Sr No.", range(1, len(pandas_df) + 1))

    # Prepare row data as list of lists
    data_rows = pandas_df.values.tolist()  # This returns list of lists

    # Prepare data for pie chart
    last_column = pandas_df.columns[-1]
    pie_data = pandas_df[last_column].value_counts(normalize=True) * 100
    pie_chart = {
        "labels": pie_data.index.tolist(),
        "values": pie_data.values.tolist()
    }

    # Prepare data for bar chart
    bar_data = pandas_df.groupby("CUSTOMER_ID")[last_column].value_counts(normalize=True).unstack(fill_value=0) * 100
    bar_chart = {
        "categories": bar_data.index.tolist(),
        "series": [
            {"name": col, "data": bar_data[col].tolist()} for col in bar_data.columns
        ]
    }

    # Return JSON response
    return JsonResponse({
        "data": data_rows,
        "pie_chart": pie_chart,
        "bar_chart": bar_chart
    })

def userLogout(request):
    try:
      del request.session['username']
    except:
      pass
    logout(request)
    return HttpResponseRedirect('/')

def prediction_button(request, id):
    data_file = get_object_or_404(DataFileUpload, pk=id)
    file_path = data_file.actual_file.path  # Full path to uploaded file

    spark = SparkSession.builder \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    df = spark.read.csv(file_path, header=True, inferSchema=True)

    pandas_df, bar_chart = run_prediction(df)

    return render(request, 'api/fraud_detection.html', {
        'id': id,
        'table_data': pandas_df.to_dict(orient="records"),
        'columns': pandas_df.columns.tolist(),
        'bar_chart': bar_chart
    })

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
