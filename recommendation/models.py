"""
Django models for SmartBiz AI recommendation system.
Used for recording recommendation history, business profiles, and tracking ML system state.
"""

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class BusinessProfile(models.Model):
    """Stores registered business and admin profile details."""

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='business_profile')
    business_name = models.CharField(max_length=255, default='Example Fashion Store')
    admin_name = models.CharField(max_length=255, default='Business Owner')
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        verbose_name = 'Business Profile'
        verbose_name_plural = 'Business Profiles'

    def __str__(self):
        return f"{self.business_name} ({self.admin_name})"


@receiver(post_save, sender=User)
def ensure_business_profile(sender, instance, created, **kwargs):
    """Ensure every User has a BusinessProfile."""
    if created:
        BusinessProfile.objects.get_or_create(
            user=instance,
            defaults={
                'business_name': 'Example Fashion Store',
                'admin_name': instance.first_name or instance.username or 'Business Owner'
            }
        )


class RecommendationHistory(models.Model):
    """Stores ML prediction results and recommendations for real customers."""

    customer_id = models.IntegerField(default=0, db_index=True)
    customer_code = models.CharField(max_length=100, default='', db_index=True, blank=True, help_text="Customer ID string or number")
    country = models.CharField(max_length=100, default='United Kingdom')
    recency = models.IntegerField(default=0, help_text="Days since last purchase")
    frequency = models.IntegerField(default=0, help_text="Total invoice count")
    monetary = models.FloatField(default=0.0, help_text="Total spend (£)")
    segment = models.CharField(max_length=100, default='General', help_text="K-Means Cluster Segment")
    cluster_id = models.IntegerField(default=0)
    recommended_products = models.JSONField(default=list, help_text="List of recommended products")
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Recommendation Histories'

    def __str__(self):
        display_id = self.customer_code or f"Customer {self.customer_id}"
        return f"{display_id} -> {self.segment} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"


class BusinessCustomer(models.Model):
    """Live business-side customer created by a business admin."""

    business_profile = models.ForeignKey(
        BusinessProfile, on_delete=models.CASCADE, related_name='customers', null=True, blank=True
    )
    customer_id = models.CharField(max_length=50, unique=True, db_index=True, editable=False)
    name = models.CharField(max_length=255)
    email = models.EmailField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Business Customer'
        verbose_name_plural = 'Business Customers'

    def save(self, *args, **kwargs):
        if not self.customer_id:
            last_cust = BusinessCustomer.objects.order_by('-id').first()
            next_num = (last_cust.id + 1) if last_cust else 1
            self.customer_id = f"{next_num:05d}"
            while BusinessCustomer.objects.filter(customer_id=self.customer_id).exists():
                next_num += 1
                self.customer_id = f"{next_num:05d}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.customer_id})"


class PurchaseTransaction(models.Model):
    """Permanent store for purchase transactions recorded by a business admin."""

    customer = models.ForeignKey(BusinessCustomer, on_delete=models.CASCADE, related_name='purchases')
    stock_code = models.CharField(max_length=50)
    product_description = models.CharField(max_length=255)
    quantity = models.IntegerField(default=1)
    unit_price = models.FloatField(default=0.0)
    purchase_date = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-purchase_date', '-created_at']
        verbose_name = 'Purchase Transaction'
        verbose_name_plural = 'Purchase Transactions'

    @property
    def total_amount(self):
        return round(self.quantity * self.unit_price, 2)

    def __str__(self):
        return f"{self.customer.customer_id} - {self.product_description} (x{self.quantity})"

