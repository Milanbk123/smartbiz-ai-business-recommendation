"""Admin configuration for SmartBiz AI."""

from django.contrib import admin
from .models import RecommendationHistory, BusinessProfile, BusinessCustomer, PurchaseTransaction


@admin.register(BusinessProfile)
class BusinessProfileAdmin(admin.ModelAdmin):
    list_display = ['business_name', 'admin_name', 'user', 'created_at']
    search_fields = ['business_name', 'admin_name', 'user__email']


@admin.register(RecommendationHistory)
class RecommendationHistoryAdmin(admin.ModelAdmin):
    list_display = ['customer_id', 'customer_code', 'country', 'segment', 'recency', 'frequency', 'monetary', 'created_at']
    list_filter = ['segment', 'country']
    search_fields = ['customer_id', 'customer_code', 'segment', 'country']
    ordering = ['-created_at']


@admin.register(BusinessCustomer)
class BusinessCustomerAdmin(admin.ModelAdmin):
    list_display = ['customer_id', 'name', 'email', 'business_profile', 'created_at']
    search_fields = ['customer_id', 'name', 'email']
    list_filter = ['created_at']


@admin.register(PurchaseTransaction)
class PurchaseTransactionAdmin(admin.ModelAdmin):
    list_display = ['customer', 'stock_code', 'product_description', 'quantity', 'unit_price', 'total_amount', 'purchase_date']
    search_fields = ['customer__customer_id', 'customer__name', 'stock_code', 'product_description']
    list_filter = ['purchase_date']

