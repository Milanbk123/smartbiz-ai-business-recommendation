"""
Recommender Engine & Inference Service
Consumes persistent artifacts trained in ml/pipeline.py.
Handles customer lookup, cluster prediction, and actual product recommendations.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CLUSTERED_RFM_PATH = os.path.join(BASE_DIR, 'dataset', 'customer_rfm_clustered.csv')
CLEANED_TRANSACTIONS_PATH = os.path.join(BASE_DIR, 'dataset', 'cleaned_transactions.csv')
TOP_PRODUCTS_PATH = os.path.join(MODELS_DIR, 'top_products_per_cluster.json')
METADATA_PATH = os.path.join(MODELS_DIR, 'metadata.json')

_SCALER = None
_KMEANS = None
_METADATA = None
_TOP_PRODUCTS = None
_RFM_DF = None
_TRANSACTIONS_DF = None


def load_artifacts():
    global _SCALER, _KMEANS, _METADATA, _TOP_PRODUCTS, _RFM_DF, _TRANSACTIONS_DF

    if _SCALER is None and os.path.exists(os.path.join(MODELS_DIR, 'scaler.pkl')):
        _SCALER = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

    if _KMEANS is None and os.path.exists(os.path.join(MODELS_DIR, 'kmeans.pkl')):
        _KMEANS = joblib.load(os.path.join(MODELS_DIR, 'kmeans.pkl'))

    if _METADATA is None and os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            _METADATA = json.load(f)

    if _TOP_PRODUCTS is None and os.path.exists(TOP_PRODUCTS_PATH):
        with open(TOP_PRODUCTS_PATH, 'r') as f:
            _TOP_PRODUCTS = json.load(f)

    if _RFM_DF is None and os.path.exists(CLUSTERED_RFM_PATH):
        _RFM_DF = pd.read_csv(CLUSTERED_RFM_PATH)

    if _TRANSACTIONS_DF is None and os.path.exists(CLEANED_TRANSACTIONS_PATH):
        _TRANSACTIONS_DF = pd.read_csv(CLEANED_TRANSACTIONS_PATH)


def get_customer_history(customer_id):
    load_artifacts()
    if _TRANSACTIONS_DF is None:
        return []

    cust_tx = _TRANSACTIONS_DF[_TRANSACTIONS_DF['CustomerID'] == int(customer_id)]
    if cust_tx.empty:
        return []

    history = (
        cust_tx.groupby(['StockCode', 'Description'])
        .agg(
            total_quantity=('Quantity', 'sum'),
            total_spend=('TotalAmount', 'sum'),
            order_count=('InvoiceNo', 'nunique'),
            last_date=('InvoiceDate', 'max')
        )
        .reset_index()
        .sort_values(by='total_spend', ascending=False)
    )

    result = []
    for _, row in history.iterrows():
        result.append({
            'stock_code': str(row['StockCode']),
            'description': str(row['Description']),
            'total_quantity': int(row['total_quantity']),
            'total_spend': round(float(row['total_spend']), 2),
            'order_count': int(row['order_count']),
            'last_date': str(row['last_date'])
        })
    return result


def recommend_for_customer(customer_id, top_n=5, exclude_bought=True):
    """
    Phase 5: Generate product recommendations for an existing Customer ID.
    1. Look up customer's cluster and segment.
    2. Retrieve customer's purchase history.
    3. Exclude products already purchased (if exclude_bought=True).
    4. Return top actual products from their cluster.
    """
    load_artifacts()

    if _RFM_DF is None:
        raise RuntimeError("Model artifacts not trained yet. Run ml/pipeline.py first.")

    cust_row = _RFM_DF[_RFM_DF['CustomerID'] == int(customer_id)]
    if cust_row.empty:
        return {'error': f'Customer ID {customer_id} not found in database.'}

    cust_data = cust_row.iloc[0].to_dict()
    cluster_id = str(cust_data['Cluster'])
    segment = cust_data['Segment']

    # Get purchase history (stock codes bought)
    history = get_customer_history(customer_id)
    bought_stock_codes = set(item['stock_code'] for item in history)

    # Get cluster top products
    cluster_products = _TOP_PRODUCTS.get(cluster_id, [])

    recommendations = []
    for prod in cluster_products:
        if exclude_bought and prod['stock_code'] in bought_stock_codes:
            continue
        recommendations.append(prod)
        if len(recommendations) >= top_n:
            break

    # If all items were filtered out, fall back to top cluster items
    if not recommendations:
        recommendations = cluster_products[:top_n]

    return {
        'customer_id': int(customer_id),
        'recency': int(cust_data['Recency']),
        'frequency': int(cust_data['Frequency']),
        'monetary': float(cust_data['Monetary']),
        'cluster_id': int(cluster_id),
        'segment': segment,
        'country': cust_data.get('PrimaryCountry', 'Unknown'),
        'purchase_history': history[:5],  # top 5 history items
        'recommendations': recommendations
    }


def predict_segment_and_recommend(recency, frequency, monetary, top_n=5):
    """
    Predict segment and recommendations for ad-hoc RFM inputs.
    """
    load_artifacts()

    if _SCALER is None or _KMEANS is None:
        raise RuntimeError("ML model is not trained yet.")

    # Log transform + Scale
    log_rec = np.log1p(float(recency))
    log_freq = np.log1p(float(frequency))
    log_mon = np.log1p(float(monetary))

    # Use a DataFrame with named columns to match how the scaler was trained
    input_df = pd.DataFrame(
        [[log_rec, log_freq, log_mon]],
        columns=['Recency', 'Frequency', 'Monetary']
    )
    features_scaled = _SCALER.transform(input_df)
    cluster_id = int(_KMEANS.predict(features_scaled)[0])

    segment_name = "Unknown"
    if _METADATA and 'cluster_profiles' in _METADATA:
        segment_name = _METADATA['cluster_profiles'].get(str(cluster_id), {}).get('segment_name', 'Segment ' + str(cluster_id))

    cluster_products = _TOP_PRODUCTS.get(str(cluster_id), [])[:top_n]

    return {
        'input_rfm': {
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary
        },
        'predicted_cluster_id': cluster_id,
        'predicted_segment': segment_name,
        'recommendations': cluster_products
    }


def get_popular_products(top_n=5, exclude_stock_codes=None):
    """Return top popular products across the historical dataset."""
    load_artifacts()
    exclude = exclude_stock_codes or set()

    popular_list = []
    if _TOP_PRODUCTS:
        seen = set()
        for cluster_id in sorted(_TOP_PRODUCTS.keys()):
            for item in _TOP_PRODUCTS[cluster_id]:
                sc = item['stock_code']
                if sc not in seen and sc not in exclude:
                    seen.add(sc)
                    popular_list.append(item)
                    if len(popular_list) >= top_n:
                        break
            if len(popular_list) >= top_n:
                break
    return popular_list[:top_n]


def get_product_catalog(search_query='', limit=30):
    """
    Returns a list of real products from the dataset for dropdown/autocomplete when recording purchases.
    """
    load_artifacts()
    query = search_query.strip().lower()

    catalog = []
    seen = set()

    if _TOP_PRODUCTS:
        for cid, prods in _TOP_PRODUCTS.items():
            for p in prods:
                sc = str(p['stock_code'])
                desc = str(p['description'])
                if sc in seen:
                    continue
                if not query or query in sc.lower() or query in desc.lower():
                    seen.add(sc)
                    catalog.append({
                        'stock_code': sc,
                        'description': desc,
                        'unit_price': float(p.get('unit_price', 1.95))
                    })
                    if len(catalog) >= limit:
                        return catalog

    if _TRANSACTIONS_DF is not None and len(catalog) < limit:
        sub_df = _TRANSACTIONS_DF
        if query:
            sub_df = sub_df[
                sub_df['StockCode'].astype(str).str.lower().str.contains(query, na=False) |
                sub_df['Description'].astype(str).str.lower().str.contains(query, na=False)
            ]
        unique_prods = sub_df[['StockCode', 'Description', 'UnitPrice']].drop_duplicates(subset=['StockCode']).head(limit)
        for _, row in unique_prods.iterrows():
            sc = str(row['StockCode'])
            if sc not in seen:
                seen.add(sc)
                catalog.append({
                    'stock_code': sc,
                    'description': str(row['Description']),
                    'unit_price': round(float(row['UnitPrice']), 2)
                })
                if len(catalog) >= limit:
                    break

    return catalog


def recommend_for_business_customer(business_customer, top_n=5):
    """
    Automatic RFM calculation, cold-start handling, and recommendation generation
    for a live BusinessCustomer stored in the Django database.
    """
    load_artifacts()
    from django.utils import timezone

    purchases = business_customer.purchases.all().order_by('purchase_date')
    purchase_count = purchases.count()

    # 1. Cold Start: No Purchase History
    if purchase_count == 0:
        popular_recs = get_popular_products(top_n=top_n)
        return {
            'customer_id': business_customer.customer_id,
            'name': business_customer.name,
            'email': business_customer.email or '',
            'has_history': False,
            'recency': None,
            'frequency': 0,
            'monetary': 0.0,
            'total_quantity': 0,
            'avg_order_value': 0.0,
            'unique_products': 0,
            'segment': 'New Customer — No Purchase History',
            'badge_class': 'bg-secondary text-white',
            'cold_start': True,
            'recommendation_label': 'Popular Products for New Customers',
            'recommendations': popular_recs,
            'purchase_history': []
        }

    # 2. RFM Calculation
    total_spending = sum(p.total_amount for p in purchases)
    total_qty = sum(p.quantity for p in purchases)
    bought_stock_codes = set(p.stock_code for p in purchases)
    unique_prods_count = len(bought_stock_codes)

    latest_purchase_date = purchases.last().purchase_date
    now = timezone.now()
    days_recency = max(0, (now - latest_purchase_date).days)
    freq = purchase_count

    avg_order_val = round(total_spending / max(1, freq), 2)

    # Convert purchases to profile history items
    purchase_history = []
    for p in purchases.order_by('-purchase_date'):
        purchase_history.append({
            'date': p.purchase_date,
            'stock_code': p.stock_code,
            'description': p.product_description,
            'quantity': p.quantity,
            'unit_price': p.unit_price,
            'total_amount': p.total_amount
        })

    # 3. Model Inference using Scaler & K-Means
    is_limited_history = freq < 2
    ml_res = predict_segment_and_recommend(
        recency=days_recency,
        frequency=freq,
        monetary=total_spending,
        top_n=top_n * 2
    )

    cluster_id = ml_res['predicted_cluster_id']
    base_segment = ml_res['predicted_segment']

    if is_limited_history:
        segment_display = "Limited Purchase History"
        badge_class = "bg-warning text-dark"
        rec_label = "Popular & Relevant Products (Limited History)"
    else:
        segment_display = base_segment
        badge_class = "bg-high-value" if "High-Value" in base_segment else "bg-low-activity"
        rec_label = f"Personalized Recommendations ({base_segment})"

    # Filter out already purchased products
    raw_cluster_products = _TOP_PRODUCTS.get(str(cluster_id), []) if _TOP_PRODUCTS else []
    filtered_recs = []
    for prod in raw_cluster_products:
        if prod['stock_code'] not in bought_stock_codes:
            filtered_recs.append(prod)
        if len(filtered_recs) >= top_n:
            break

    # Fallback to popular if needed
    if len(filtered_recs) < top_n:
        pop_recs = get_popular_products(top_n=top_n, exclude_stock_codes=bought_stock_codes)
        for pr in pop_recs:
            if pr['stock_code'] not in set(r['stock_code'] for r in filtered_recs):
                filtered_recs.append(pr)
            if len(filtered_recs) >= top_n:
                break

    return {
        'customer_id': business_customer.customer_id,
        'name': business_customer.name,
        'email': business_customer.email or '',
        'has_history': True,
        'recency': days_recency,
        'frequency': freq,
        'monetary': round(total_spending, 2),
        'total_quantity': total_qty,
        'avg_order_value': avg_order_val,
        'unique_products': unique_prods_count,
        'cluster_id': cluster_id,
        'segment': segment_display,
        'base_segment': base_segment,
        'badge_class': badge_class,
        'cold_start': is_limited_history,
        'recommendation_label': rec_label,
        'recommendations': filtered_recs[:top_n],
        'purchase_history': purchase_history
    }

