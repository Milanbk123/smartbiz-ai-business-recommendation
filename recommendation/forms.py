"""
Django forms for SmartBiz AI.
Includes business registration, email authentication, and customer recommendation lookup forms.
"""

from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from .models import BusinessCustomer, PurchaseTransaction



class RegistrationForm(forms.Form):
    """
    Business Registration Form.
    Contains ONLY:
    1. Business Name
    2. Admin Name
    3. Email
    4. Password
    """

    business_name = forms.CharField(
        max_length=255,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Example Fashion Store',
            'id': 'id_business_name',
            'autocomplete': 'organization'
        }),
        error_messages={'required': 'Business Name is required.'}
    )

    admin_name = forms.CharField(
        max_length=255,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Business Owner',
            'id': 'id_admin_name',
            'autocomplete': 'name'
        }),
        error_messages={'required': 'Admin Name is required.'}
    )

    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'owner@example.com',
            'id': 'id_email',
            'autocomplete': 'email'
        }),
        error_messages={
            'required': 'Email address is required.',
            'invalid': 'Please enter a valid email address.'
        }
    )

    password = forms.CharField(
        min_length=6,
        required=True,
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': '********',
            'id': 'id_password',
            'autocomplete': 'new-password'
        }),
        error_messages={
            'required': 'Password is required.',
            'min_length': 'Password must be at least 6 characters long.'
        }
    )

    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip().lower()
        if User.objects.filter(email__iexact=email).exists() or User.objects.filter(username__iexact=email).exists():
            raise ValidationError("An account with this email address already exists. Please sign in.")
        return email


class EmailLoginForm(forms.Form):
    """
    Login Form using Email and Password.
    """

    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'owner@example.com',
            'id': 'id_login_email',
            'autocomplete': 'email'
        }),
        error_messages={
            'required': 'Email address is required.',
            'invalid': 'Please enter a valid email address.'
        }
    )

    password = forms.CharField(
        required=True,
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': '********',
            'id': 'id_login_password',
            'autocomplete': 'current-password'
        }),
        error_messages={'required': 'Password is required.'}
    )


class CustomerLookupForm(forms.Form):
    """Form to lookup recommendations by Customer ID (Live Business or Dataset)."""

    customer_id = forms.CharField(
        max_length=100,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter Customer ID (e.g. 17850, CUST-00001, or 1)',
            'id': 'customer_id_input'
        }),
        error_messages={
            'required': 'Customer ID is required.'
        }
    )
    top_n = forms.IntegerField(
        initial=5,
        min_value=1,
        max_value=20,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '5'}),
        required=False
    )


class AdHocRFMForm(forms.Form):
    """Form to calculate recommendations for ad-hoc RFM metrics."""

    recency = forms.IntegerField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Days since last purchase (e.g. 15)'})
    )
    frequency = forms.IntegerField(
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Total orders placed (e.g. 8)'})
    )
    monetary = forms.FloatField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Total spend in £ (e.g. 2500)'})
    )


class CustomerFilterForm(forms.Form):
    """Form for filtering customer list by search term, country, or segment."""

    search = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Search Customer ID or Country...'})
    )
    segment = forms.ChoiceField(
        required=False,
        choices=[('', 'All Segments')],
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    def __init__(self, *args, segments=None, **kwargs):
        super().__init__(*args, **kwargs)
        if segments:
            self.fields['segment'].choices = [('', 'All Segments')] + [(s, s) for s in segments]


class AddBusinessCustomerForm(forms.ModelForm):
    """Form to create a new BusinessCustomer."""

    class Meta:
        model = BusinessCustomer
        fields = ['name', 'email']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Customer Name (e.g. Rahul Kumar)',
                'id': 'id_customer_name'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'rahul@example.com (optional)',
                'id': 'id_customer_email'
            }),
        }
        error_messages = {
            'name': {'required': 'Customer name is required.'}
        }

    def clean_name(self):
        name = self.cleaned_data.get('name', '').strip()
        if not name:
            raise ValidationError('Customer name is required.')
        return name


class RecordPurchaseForm(forms.ModelForm):
    """Form to record a purchase transaction for a BusinessCustomer."""

    class Meta:
        model = PurchaseTransaction
        fields = ['customer', 'stock_code', 'product_description', 'quantity', 'unit_price', 'purchase_date']
        widgets = {
            'customer': forms.Select(attrs={'class': 'form-select', 'id': 'id_select_customer'}),
            'stock_code': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 85123A', 'id': 'id_stock_code'}),
            'product_description': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Product Description', 'id': 'id_product_description'}),
            'quantity': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '1', 'min': '1', 'id': 'id_quantity'}),
            'unit_price': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '0.00', 'step': '0.01', 'min': '0', 'id': 'id_unit_price'}),
            'purchase_date': forms.DateTimeInput(attrs={'class': 'form-control', 'type': 'datetime-local', 'id': 'id_purchase_date'}),
        }

    def __init__(self, *args, business_profile=None, **kwargs):
        super().__init__(*args, **kwargs)
        from django.utils import timezone
        from .models import BusinessCustomer

        if business_profile:
            self.fields['customer'].queryset = BusinessCustomer.objects.filter(business_profile=business_profile)
        else:
            self.fields['customer'].queryset = BusinessCustomer.objects.all()

        self.fields['purchase_date'].required = False
        if not self.initial.get('purchase_date'):
            self.initial['purchase_date'] = timezone.now().strftime('%Y-%m-%dT%H:%M')

    def clean_purchase_date(self):
        from django.utils import timezone
        pdate = self.cleaned_data.get('purchase_date')
        if not pdate:
            return timezone.now()
        return pdate

    def clean_quantity(self):
        qty = self.cleaned_data.get('quantity')
        if qty is None or qty <= 0:
            raise ValidationError('Quantity must be greater than 0.')
        return qty

    def clean_unit_price(self):
        price = self.cleaned_data.get('unit_price')
        if price is None or price < 0:
            raise ValidationError('Unit price cannot be negative.')
        return price

