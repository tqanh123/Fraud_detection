from django.contrib import admin
from django.urls import path,include
from .views import base,delete_data,upload_data,view_data,retrieve_data_by_id,change_password,login2,account_details,about,dashboard,userLogout,reports,upload_credit_data, prediction_button

urlpatterns = [
    path('',base),
    path('login/',login2,name='login2'),
    path('logout/',userLogout,name='userLogout'),
    path('about/',about,name='about'),
    path('dashboard/',dashboard,name='dashboard'),
    path('reports/',reports,name='reports'),
    path('upload_credit_data/',upload_credit_data,name='upload_credit_data'),
    
    #for main adminstrator upload 
    
    path('upload_data/',upload_data,name='upload_data'),
    path('delete_data/<int:id>/',delete_data,name='delete_data'),

    path('account_details/',account_details,name='account_details'),
    path('change_password/',change_password,name='change_password'),
    path('view_data/<int:id>',view_data,name='view_data'),
    path('retrieve/<int:id>/',retrieve_data_by_id,name='retrieve_data_by_id'),
    path('predict_button/<int:id>/',prediction_button,name='predict_button'),
]
