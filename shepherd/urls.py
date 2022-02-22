from django.urls import path

from . import views

urlpatterns = [
    path('login_user/', views.login_user, name='login_user'),
    path('env/', views.env, name='env'),
    path('send_curve/', views.send_curve, name='send_curve'),
    path('generate_zip/', views.generate_zip, name='generate_zip'),
    path('delete_zip/', views.delete_zip, name='delete_zip'),
    path('delete_curve/', views.delete_curve, name='delete_curve'),
    path('export_curve_CSV/', views.export_curve_CSV, name='export_curve_CSV'),
    path('kill_processes/', views.kill_processes, name='kill_processes'),
]
