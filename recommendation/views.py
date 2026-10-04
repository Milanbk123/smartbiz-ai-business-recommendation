"""
Views for SmartBiz AI recommendation system.
Integrates existing ML model trained on Online Retail.xlsx dataset.
Protected with Django authentication.
"""

import os
import json
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator

from django.db import models
from .models import RecommendationHistory, BusinessProfile, BusinessCustomer, PurchaseTransaction
from .forms import (
    RegistrationForm, EmailLoginForm, CustomerLookupForm, AdHocRFMForm, CustomerFilterForm,
    AddBusinessCustomerForm, RecordPurchaseForm
)
from ml.recommender import (
    recommend_for_customer,
    predict_segment_and_recommend,
    get_customer_history,
    load_artifacts,
    get_product_catalog,
    recommend_for_business_customer
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTERED_RFM_PATH = os.path.join(BASE_DIR, 'dataset', 'customer_rfm_clustered.csv')
CLEANED_TRANSACTIONS_PATH = os.path.join(BASE_DIR, 'dataset', 'cleaned_transactions.csv')
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'metadata.json')


def _load_rfm_data():
    if os.path.exists(CLUSTERED_RFM_PATH):
        return pd.read_csv(CLUSTERED_RFM_PATH)
    return pd.DataFrame()


def _load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    return {}


# ─── Auth Views ──────────────────────────────────────────────────────────────

def register_view(request):
    """Business Registration Page."""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            business_name = form.cleaned_data['business_name'].strip()
            admin_name = form.cleaned_data['admin_name'].strip()
            email = form.cleaned_data['email'].strip().lower()
            password = form.cleaned_data['password']

            # Create User using Django password hashing securely
            user = User.objects.create_user(
                username=email,
                email=email,
                password=password,
                first_name=admin_name
            )

            # Store Business Profile
            BusinessProfile.objects.update_or_create(
                user=user,
                defaults={
                    'business_name': business_name,
                    'admin_name': admin_name
                }
            )

            # Log user in
            login(request, user, backend='recommendation.backends.EmailBackend')
            messages.success(request, f"Welcome to SmartBiz AI, {admin_name}! Your business profile ({business_name}) was successfully created.")
            return redirect('dashboard')
        else:
            messages.error(request, "Registration failed. Please correct the errors below.")
    else:
        form = RegistrationForm()

    return render(request, 'register.html', {'form': form})


def login_view(request):
    """Email + Password Login Page."""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = EmailLoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email'].strip().lower()
            password = form.cleaned_data['password']

            user = authenticate(request, username=email, password=password)
            if user is not None:
                login(request, user)
                profile = getattr(user, 'business_profile', None)
                admin_name = profile.admin_name if profile else (user.first_name or user.username)
                messages.success(request, f"Welcome back, {admin_name}!")
                next_url = request.GET.get('next', 'dashboard')
                return redirect(next_url)
            else:
                messages.error(request, "Invalid email or password. Please try again.")
        else:
            messages.error(request, "Please enter a valid email address and password.")
    else:
        form = EmailLoginForm()

    return render(request, 'login.html', {'form': form})


def logout_view(request):
    """Logout endpoint."""
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('login')


# ─── Dashboard ───────────────────────────────────────────────────────────────

@login_required(login_url='login')
def dashboard(request):
    """Admin Dashboard with statistics calculated from Online Retail dataset."""
    rfm_df = _load_rfm_data()
    meta = _load_metadata()

    total_customers = len(rfm_df) if not rfm_df.empty else 4338
    total_transactions = meta.get('cleaned_rows', 392692)
    total_revenue = float(rfm_df['Monetary'].sum()) if not rfm_df.empty else 8911407.90
    total_recommendations = RecommendationHistory.objects.count()

    # Product count
    total_products = 3897
    if os.path.exists(CLEANED_TRANSACTIONS_PATH):
        try:
            tx_df = pd.read_csv(CLEANED_TRANSACTIONS_PATH, usecols=['StockCode'])
            total_products = tx_df['StockCode'].nunique()
        except Exception:
            pass

    # Customer Segments summary
    segment_counts = {}
    segment_labels = []
    segment_values = []
    if not rfm_df.empty and 'Segment' in rfm_df.columns:
        seg_summary = rfm_df['Segment'].value_counts()
        segment_labels = list(seg_summary.index)
        segment_values = [int(v) for v in seg_summary.values]
        for seg, count in seg_summary.items():
            segment_counts[seg] = {
                'count': int(count),
                'percent': round((count / total_customers) * 100, 1)
            }

    # Recent Recommendations from DB
    recent_recs = RecommendationHistory.objects.all()[:6]

    context = {
        'page': 'dashboard',
        'total_customers': f"{total_customers:,}",
        'total_transactions': f"{total_transactions:,}",
        'total_revenue': f"{total_revenue:,.2f}",
        'total_products': f"{total_products:,}",
        'total_recommendations': f"{total_recommendations:,}",
        'segment_counts': segment_counts,
        'segment_labels_json': json.dumps(segment_labels),
        'segment_values_json': json.dumps(segment_values),
        'recent_recs': recent_recs,
    }
    return render(request, 'dashboard.html', context)


# ─── Get Prediction Page ─────────────────────────────────────────────────────

@login_required(login_url='login')
def prediction(request):
    """Customer recommendation workflow using real dataset models."""
    rfm_df = _load_rfm_data()
    sample_customers = []

    if not rfm_df.empty:
        # Top 15 representative customer IDs for quick testing
        sample_customers = rfm_df.sort_values(by='Monetary', ascending=False)['CustomerID'].head(15).tolist()

    result = None
    customer_info = None

    if request.method == 'POST':
        form_type = request.POST.get('form_type', 'lookup')

        if form_type == 'lookup':
            lookup_form = CustomerLookupForm(request.POST)
            adhoc_form = AdHocRFMForm()

            if lookup_form.is_valid():
                raw_id = str(lookup_form.cleaned_data['customer_id']).strip()
                top_n = lookup_form.cleaned_data.get('top_n') or 5
                profile = getattr(request.user, 'business_profile', None)

                # 1. Try Live Business Customer lookup
                biz_cust = BusinessCustomer.objects.filter(
                    business_profile=profile, customer_id__iexact=raw_id
                ).first()

                if not biz_cust and raw_id.isdigit():
                    biz_cust = BusinessCustomer.objects.filter(
                        business_profile=profile, customer_id=raw_id.zfill(5)
                    ).first()

                if not biz_cust and raw_id.isdigit():
                    biz_cust = BusinessCustomer.objects.filter(
                        business_profile=profile, id=int(raw_id)
                    ).first()

                if biz_cust:
                    try:
                        rec_res = recommend_for_business_customer(biz_cust, top_n=top_n)
                        result = {
                            'cluster_id': rec_res.get('cluster_id', 0),
                            'recommendations': rec_res['recommendations']
                        }
                        customer_info = {
                            'customer_id': biz_cust.customer_id,
                            'country': 'United Kingdom',
                            'recency': rec_res['recency'] if rec_res['recency'] is not None else 0,
                            'frequency': rec_res['frequency'],
                            'monetary': rec_res['monetary'],
                            'total_quantity': rec_res['total_quantity'],
                            'avg_order_value': rec_res['avg_order_value'],
                            'unique_products': rec_res['unique_products'],
                            'segment': rec_res['segment'],
                        }
                        RecommendationHistory.objects.create(
                            customer_id=0,
                            customer_code=biz_cust.customer_id,
                            country='United Kingdom',
                            recency=customer_info['recency'],
                            frequency=customer_info['frequency'],
                            monetary=customer_info['monetary'],
                            segment=customer_info['segment'],
                            cluster_id=rec_res.get('cluster_id', 0),
                            recommended_products=rec_res['recommendations']
                        )
                        messages.success(request, f"Recommendations generated for Live Customer {biz_cust.name} ({biz_cust.customer_id}).")
                    except Exception as e:
                        messages.error(request, f"Prediction error: {str(e)}")

                # 2. Try Historical Dataset Customer lookup
                elif raw_id.isdigit() and not rfm_df.empty and int(raw_id) in rfm_df['CustomerID'].values:
                    try:
                        cid = int(raw_id)
                        res = recommend_for_customer(cid, top_n=top_n)
                        if 'error' in res:
                            messages.error(request, res['error'])
                        else:
                            result = res
                            cust_row = rfm_df[rfm_df['CustomerID'] == cid].iloc[0]
                            customer_info = {
                                'customer_id': cid,
                                'country': cust_row.get('PrimaryCountry', 'United Kingdom'),
                                'recency': int(cust_row['Recency']),
                                'frequency': int(cust_row['Frequency']),
                                'monetary': float(cust_row['Monetary']),
                                'total_quantity': int(cust_row['TotalQuantity']),
                                'avg_order_value': float(cust_row['AvgOrderValue']),
                                'unique_products': int(cust_row['UniqueProducts']),
                                'segment': cust_row['Segment'],
                            }
                            RecommendationHistory.objects.create(
                                customer_id=cid,
                                country=customer_info['country'],
                                recency=customer_info['recency'],
                                frequency=customer_info['frequency'],
                                monetary=customer_info['monetary'],
                                segment=customer_info['segment'],
                                cluster_id=res['cluster_id'],
                                recommended_products=res['recommendations']
                            )
                            messages.success(request, f"Recommendations generated for Customer #{cid}.")
                    except Exception as e:
                        messages.error(request, f"Prediction error: {str(e)}")

                # 3. New Customer / Cold Start lookup
                else:
                    try:
                        from ml.recommender import get_popular_products
                        pop_recs = get_popular_products(top_n=top_n)
                        result = {
                            'cluster_id': 0,
                            'recommendations': pop_recs
                        }
                        customer_info = {
                            'customer_id': raw_id,
                            'country': 'United Kingdom',
                            'recency': 0,
                            'frequency': 0,
                            'monetary': 0.0,
                            'total_quantity': 0,
                            'avg_order_value': 0.0,
                            'unique_products': 0,
                            'segment': 'New Customer — Cold Start Recommendations',
                        }
                        RecommendationHistory.objects.create(
                            customer_id=int(raw_id) if raw_id.isdigit() else 0,
                            customer_code=raw_id if not raw_id.isdigit() else '',
                            country='United Kingdom',
                            recency=0,
                            frequency=0,
                            monetary=0.0,
                            segment=customer_info['segment'],
                            cluster_id=0,
                            recommended_products=pop_recs
                        )
                        messages.success(request, f"Generated recommendations for New Customer #{raw_id}.")
                    except Exception as e:
                        messages.error(request, f"Prediction error: {str(e)}")

        elif form_type == 'adhoc':
            lookup_form = CustomerLookupForm()
            adhoc_form = AdHocRFMForm(request.POST)

            if adhoc_form.is_valid():
                data = adhoc_form.cleaned_data
                try:
                    res = predict_segment_and_recommend(
                        recency=data['recency'],
                        frequency=data['frequency'],
                        monetary=data['monetary'],
                        top_n=5
                    )
                    result = res
                    customer_info = {
                        'customer_id': 'Ad-Hoc Input',
                        'country': 'N/A',
                        'recency': data['recency'],
                        'frequency': data['frequency'],
                        'monetary': data['monetary'],
                        'total_quantity': 'N/A',
                        'avg_order_value': round(data['monetary'] / data['frequency'], 2),
                        'unique_products': 'N/A',
                        'segment': res['predicted_segment'],
                    }
                    messages.success(request, f"Predicted segment: {res['predicted_segment']}.")
                except Exception as e:
                    messages.error(request, f"Prediction error: {str(e)}")

    else:
        lookup_form = CustomerLookupForm()
        adhoc_form = AdHocRFMForm()

    return render(request, 'prediction.html', {
        'page': 'prediction',
        'lookup_form': lookup_form,
        'adhoc_form': adhoc_form,
        'sample_customers': sample_customers,
        'result': result,
        'customer_info': customer_info,
    })


# ─── Prediction API (JSON) ────────────────────────────────────────────────────

@login_required(login_url='login')
def predict_api(request):
    """JSON API endpoint for recommendation inference."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    try:
        body = json.loads(request.body)
        customer_id = body.get('customer_id')

        if customer_id:
            res = recommend_for_customer(customer_id, top_n=body.get('top_n', 5))
            return JsonResponse(res)
        else:
            res = predict_segment_and_recommend(
                recency=body.get('recency', 30),
                frequency=body.get('frequency', 5),
                monetary=body.get('monetary', 1000),
                top_n=body.get('top_n', 5)
            )
            return JsonResponse(res)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ─── Live Business Customer & Purchase Workflow ─────────────────────────────────

@login_required(login_url='login')
def add_customer(request):
    """Create a new Business Customer."""
    profile = getattr(request.user, 'business_profile', None)

    if request.method == 'POST':
        form = AddBusinessCustomerForm(request.POST)
        if form.is_valid():
            customer = form.save(commit=False)
            customer.business_profile = profile
            customer.save()
            messages.success(request, f"Customer {customer.name} ({customer.customer_id}) created successfully.")
            return render(request, 'add_customer_success.html', {
                'page': 'add_customer',
                'customer': customer
            })
        else:
            messages.error(request, "Please correct the errors in the customer form.")
    else:
        form = AddBusinessCustomerForm()

    return render(request, 'add_customer.html', {
        'page': 'add_customer',
        'form': form
    })


@login_required(login_url='login')
def record_purchase(request):
    """Record a purchase transaction for a live Business Customer."""
    profile = getattr(request.user, 'business_profile', None)
    initial_customer_id = request.GET.get('customer_id') or request.GET.get('customer')

    initial_cust = None
    if initial_customer_id:
        initial_cust = BusinessCustomer.objects.filter(
            business_profile=profile, customer_id=initial_customer_id
        ).first()

    recorded_transaction = None

    if request.method == 'POST':
        form = RecordPurchaseForm(request.POST, business_profile=profile)
        if form.is_valid():
            transaction = form.save()
            messages.success(request, f"Purchase recorded successfully for {transaction.customer.name}.")

            # Also generate recommendation to save to history
            try:
                rec_res = recommend_for_business_customer(transaction.customer, top_n=5)
                RecommendationHistory.objects.create(
                    customer_id=0,
                    customer_code=transaction.customer.customer_id,
                    country='United Kingdom',
                    recency=rec_res['recency'] or 0,
                    frequency=rec_res['frequency'],
                    monetary=rec_res['monetary'],
                    segment=rec_res['segment'],
                    cluster_id=rec_res.get('cluster_id', 0),
                    recommended_products=rec_res['recommendations']
                )
            except Exception:
                pass

            recorded_transaction = transaction
            # Clear form for potential next entry
            form = RecordPurchaseForm(business_profile=profile)
        else:
            messages.error(request, "Please correct the errors in the purchase form.")
    else:
        form = RecordPurchaseForm(
            business_profile=profile,
            initial={'customer': initial_cust.pk if initial_cust else None}
        )

    product_suggestions = get_product_catalog(limit=30)

    return render(request, 'record_purchase.html', {
        'page': 'record_purchase',
        'form': form,
        'recorded_transaction': recorded_transaction,
        'product_suggestions': product_suggestions,
        'selected_customer': initial_cust
    })


@login_required(login_url='login')
def api_product_search(request):
    """JSON API endpoint for real product catalog search."""
    q = request.GET.get('q', '').strip()
    products = get_product_catalog(search_query=q, limit=20)
    return JsonResponse({'products': products})


# ─── Customers Page ───────────────────────────────────────────────────────────

@login_required(login_url='login')
def customers(request):
    """Customer management page displaying live business customers and historical dataset customers."""
    profile = getattr(request.user, 'business_profile', None)

    search = request.GET.get('search', '').strip()
    tab = request.GET.get('tab', 'historical')  # 'historical' or 'live'

    # Live Business Customers
    biz_customers_qs = BusinessCustomer.objects.filter(business_profile=profile)
    if search and tab == 'live':
        biz_customers_qs = biz_customers_qs.filter(
            models.Q(name__icontains=search) |
            models.Q(customer_id__icontains=search) |
            models.Q(email__icontains=search)
        )

    live_customers = []
    for bc in biz_customers_qs:
        purchases = bc.purchases.all()
        cnt = purchases.count()
        tot_spend = sum(p.total_amount for p in purchases)
        
        # Determine RFM / Segment tag
        rec_data = recommend_for_business_customer(bc, top_n=1)
        
        live_customers.append({
            'object': bc,
            'customer_id': bc.customer_id,
            'name': bc.name,
            'email': bc.email or 'N/A',
            'created_at': bc.created_at,
            'purchase_count': cnt,
            'total_spend': round(tot_spend, 2),
            'segment': rec_data['segment'],
            'badge_class': rec_data['badge_class']
        })

    # Historical Dataset Customers
    rfm_df = _load_rfm_data()
    segment_filter = request.GET.get('segment', '').strip()
    segments_list = list(rfm_df['Segment'].unique()) if not rfm_df.empty and 'Segment' in rfm_df.columns else []

    filtered_df = rfm_df.copy()
    if not filtered_df.empty:
        # Guarantee numeric sorting by CustomerID in ascending order
        filtered_df['CustomerID'] = pd.to_numeric(filtered_df['CustomerID'], errors='coerce')
        filtered_df = filtered_df.sort_values(by='CustomerID', ascending=True)

        if search and (tab == 'historical' or not tab):
            if search.isdigit():
                filtered_df = filtered_df[filtered_df['CustomerID'] == int(search)]
            else:
                filtered_df = filtered_df[filtered_df['PrimaryCountry'].astype(str).str.contains(search, case=False, na=False)]
        if segment_filter:
            filtered_df = filtered_df[filtered_df['Segment'] == segment_filter]

    historical_records = filtered_df.to_dict('records') if not filtered_df.empty else []
    paginator = Paginator(historical_records, 15)
    page_number = request.GET.get('page', 1)
    historical_page_obj = paginator.get_page(page_number)

    form = CustomerFilterForm(request.GET, segments=segments_list)

    return render(request, 'customers.html', {
        'page': 'customers',
        'tab': tab,
        'live_customers': live_customers,
        'live_count': len(live_customers),
        'historical_page_obj': historical_page_obj,
        'total_historical': len(historical_records),
        'form': form,
        'search': search,
        'segment_filter': segment_filter,
    })


@login_required(login_url='login')
def customer_detail(request, customer_id):
    """View details, RFM, purchase history, and recommendations for a customer (Live or Historical)."""
    profile = getattr(request.user, 'business_profile', None)

    # Case A: Live Business Customer
    biz_cust = BusinessCustomer.objects.filter(
        customer_id__iexact=customer_id
    ).first()

    if not biz_cust and str(customer_id).isdigit():
        biz_cust = BusinessCustomer.objects.filter(id=int(customer_id)).first()

    if biz_cust:
        rec_data = recommend_for_business_customer(biz_cust, top_n=5)

        return render(request, 'customer_detail.html', {
            'page': 'customers',
            'is_live': True,
            'customer': biz_cust,
            'rec_data': rec_data,
            'history': rec_data['purchase_history'],
            'recommendations': rec_data['recommendations']
        })

    # Case B: Historical Dataset Customer (e.g. Customer ID 17850)
    rfm_df = _load_rfm_data()
    if str(customer_id).isdigit():
        cid_int = int(customer_id)
        cust_rows = rfm_df[rfm_df['CustomerID'] == cid_int] if not rfm_df.empty else pd.DataFrame()
        if not cust_rows.empty:
            customer = cust_rows.iloc[0].to_dict()
            history = get_customer_history(cid_int)
            recommendations_res = recommend_for_customer(cid_int, top_n=5)

            return render(request, 'customer_detail.html', {
                'page': 'customers',
                'is_live': False,
                'customer': customer,
                'history': history,
                'recommendations': recommendations_res.get('recommendations', [])
            })

    messages.error(request, f"Customer '{customer_id}' not found.")
    return redirect('customers')


# ─── Recommendation History ───────────────────────────────────────────────────

@login_required(login_url='login')
def history(request):
    """View saved recommendation histories."""
    qs = RecommendationHistory.objects.all()

    search = request.GET.get('search', '').strip()
    if search:
        if search.isdigit():
            qs = qs.filter(models.Q(customer_id=int(search)) | models.Q(customer_code__icontains=search))
        else:
            qs = qs.filter(models.Q(customer_code__icontains=search) | models.Q(segment__icontains=search))

    paginator = Paginator(qs, 15)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    return render(request, 'history.html', {
        'page': 'history',
        'page_obj': page_obj,
        'search': search,
        'total': qs.count(),
    })



# ─── Analytics ────────────────────────────────────────────────────────────────

@login_required(login_url='login')
def analytics(request):
    """Analytics page with chart statistics calculated from Online Retail dataset."""
    rfm_df = _load_rfm_data()

    if rfm_df.empty:
        return render(request, 'analytics.html', {'page': 'analytics', 'has_data': False})

    # Segment distribution
    seg_counts = rfm_df['Segment'].value_counts()

    # Monetary by Segment (average monetary spend)
    seg_monetary = rfm_df.groupby('Segment')['Monetary'].mean().round(2)

    # Frequency by Segment (average order count)
    seg_frequency = rfm_df.groupby('Segment')['Frequency'].mean().round(1)

    # Recency by Segment (average recency days)
    seg_recency = rfm_df.groupby('Segment')['Recency'].mean().round(1)

    # Country breakdown (top 10 countries)
    country_counts = rfm_df['PrimaryCountry'].value_counts().head(10)

    context = {
        'page': 'analytics',
        'has_data': True,
        'total_customers': len(rfm_df),
        'seg_labels': json.dumps(list(seg_counts.index)),
        'seg_counts': json.dumps([int(v) for v in seg_counts.values]),

        'mon_labels': json.dumps(list(seg_monetary.index)),
        'mon_values': json.dumps([float(v) for v in seg_monetary.values]),

        'freq_labels': json.dumps(list(seg_frequency.index)),
        'freq_values': json.dumps([float(v) for v in seg_frequency.values]),

        'rec_labels': json.dumps(list(seg_recency.index)),
        'rec_values': json.dumps([float(v) for v in seg_recency.values]),

        'country_labels': json.dumps(list(country_counts.index)),
        'country_values': json.dumps([int(v) for v in country_counts.values]),
    }
    return render(request, 'analytics.html', context)


# ─── ML Model Info ────────────────────────────────────────────────────────────

@login_required(login_url='login')
def model_info(request):
    """Display trained K-Means model metadata & evaluation metrics."""
    meta = _load_metadata()

    return render(request, 'model_info.html', {
        'page': 'model_info',
        'meta': meta,
        'selected_k': meta.get('selected_k', 2),
        'silhouette_score': meta.get('best_silhouette_score', 0.4328),
        'num_customers': meta.get('num_customers', 4338),
        'cleaned_rows': meta.get('cleaned_rows', 392692),
        'features': meta.get('features_used', ['Recency', 'Frequency', 'Monetary']),
        'supporting_features': meta.get('supporting_features', ['TotalQuantity', 'AvgOrderValue', 'UniqueProducts', 'PrimaryCountry']),
        'eval_grid': meta.get('k_evaluation', []),
        'cluster_profiles': meta.get('cluster_profiles', {}),
    })


# ─── About ────────────────────────────────────────────────────────────────────

@login_required(login_url='login')
def about(request):
    """About project page."""
    return render(request, 'about.html', {
        'page': 'about',
    })
