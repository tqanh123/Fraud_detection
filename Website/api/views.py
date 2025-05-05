import io
import pickle
import pandas as pd
from django.shortcuts import render,redirect,get_object_or_404
from django.http.response import HttpResponseRedirect,HttpResponse,JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import DataFileUpload

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, date_format, count, lit


def base(request):
    return render(request,'api/landing_page.html')
    
def upload_credit_data(request):
    return render(request,'api/upload_credit_data.html')

def prediction_button(request,id):
    return render(request,'api/fraud_detection.html', {'id': id})
    
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

# def retrieve_data_by_id(request, id):
#     obj = DataFileUpload.objects.get(id=id)
#     df = pd.read_csv(obj.actual_file.path)

#     # Receive parameters from DataTables on the frontend
#     draw = int(request.GET.get('draw', 1))
#     start = int(request.GET.get('start', 0))
#     length = int(request.GET.get('length', 10))
#     customer_id = request.GET.get('customer_id')
#     tx_month = request.GET.get('tx_month')

#     print("retrive data")
#     print("draw: ", draw)
#     print("start: ", start)
#     print("customer id: ", customer_id)
#     print("tx month: ", tx_month)


#     # Filter data based on customer_id and tx_month
#     if customer_id:
#         customer_id = int(customer_id)
#         df = df[df['CUSTOMER_ID'] == customer_id]

#     if tx_month:
#         try:
#             print("tx month: ", tx_month)
#             df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
#             # Extract month and year from TX_DATETIME
#             df['TX_MONTH_YEAR'] = df['TX_DATETIME'].dt.strftime('%m-%Y')
#             # Compare with tx_month
#             df = df[df['TX_MONTH_YEAR'] == tx_month]
#         except Exception as e:
#             print("Error in filtering by month:", e)

#     # Paginate the data from the CSV using start and length
#     paginated_df = df.iloc[start:start+length].reset_index()
#     paginated_df['index'] = paginated_df['index'] + 1 + start

#     # Convert the paginated data to a list of lists
#     data = paginated_df.values.tolist()

#     # Return a JSON response suitable for DataTables
#     return JsonResponse({
#         'draw': draw,
#         'recordsTotal': len(df),
#         'recordsFiltered': len(df),
#         'data': data,
#     })
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
        "data": pandas_df.to_dict(orient="records"),
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
    

def login2(request):
    data = {}
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        print(user)
        if user:
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
