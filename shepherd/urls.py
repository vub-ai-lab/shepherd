from django.urls import path

from . import views

urlpatterns = [
    path('login_user/', views.login_user, name='login_user'),
    path('env/', views.env, name='env'),
    path('send_curve/', views.send_curve, name='send_curve'),
    path('generate_zip/', views.generate_zip, name='generate_zip'),
    path('get_how_much_time_since_last_time/', views.get_how_much_time_since_last_time, name='get_how_much_time_since_last_time')
] 
