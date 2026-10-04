"""URL patterns for SmartBiz AI recommendation app."""

from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('', views.dashboard, name='dashboard'),
    path('prediction/', views.prediction, name='prediction'),
    path('api/predict/', views.predict_api, name='predict_api'),
    path('api/products/', views.api_product_search, name='api_products'),
    path('customers/', views.customers, name='customers'),
    path('customers/add/', views.add_customer, name='add_customer'),
    path('purchases/record/', views.record_purchase, name='record_purchase'),
    path('customers/<str:customer_id>/', views.customer_detail, name='customer_detail'),
    path('history/', views.history, name='history'),
    path('analytics/', views.analytics, name='analytics'),
    path('model/', views.model_info, name='model_info'),
    path('about/', views.about, name='about'),
]

