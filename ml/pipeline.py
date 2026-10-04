"""
ML Training Pipeline for Online Retail Dataset
Phase 3: K-Means evaluation & training
Phase 4: Customer segment profiling
Phase 5: Product recommendation rules
Phase 6: Persistence of models and outputs for Django consumption
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .data_processor import load_and_clean_data, compute_rfm_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CLUSTERED_RFM_PATH = os.path.join(BASE_DIR, 'dataset', 'customer_rfm_clustered.csv')
TOP_PRODUCTS_PATH = os.path.join(MODELS_DIR, 'top_products_per_cluster.json')
METADATA_PATH = os.path.join(MODELS_DIR, 'metadata.json')


def train_pipeline(force_reprocess=False):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Phase 1 & 2: Load clean data and compute RFM
    clean_df = load_and_clean_data(force_reprocess=force_reprocess)
    rfm_df = compute_rfm_features(clean_df, force_reprocess=force_reprocess)

    # 2. Log transformation to handle right-skewness in RFM
    rfm_log = np.log1p(rfm_df[['Recency', 'Frequency', 'Monetary']])

    # 3. Standard Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # 4. Phase 3: Evaluate K for K-Means (K from 2 to 8)
    evaluation_results = []
    best_k = 4
    best_silhouette = -1
    best_model = None

    print("\n" + "=" * 60)
    print("PHASE 3: K-MEANS CLUSTERING EVALUATION")
    print("=" * 60)
    print(f"{'K':<5} | {'Inertia (SSE)':<18} | {'Silhouette Score':<18}")
    print("-" * 50)

    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(rfm_scaled)
        sil_score = silhouette_score(rfm_scaled, labels)
        inertia = kmeans.inertia_

        evaluation_results.append({
            'k': int(k),
            'inertia': float(round(inertia, 2)),
            'silhouette_score': float(round(sil_score, 4))
        })

        print(f"{k:<5} | {inertia:<18.2f} | {sil_score:<18.4f}")

        if sil_score > best_silhouette:
            best_silhouette = sil_score
            best_k = k
            best_model = kmeans

    print("-" * 50)
    print(f"[K-MEANS] Selected K = {best_k} (Highest Silhouette Score: {best_silhouette:.4f})")

    # Fit best model explicitly to ensure final labels
    final_labels = best_model.labels_
    rfm_df['Cluster'] = final_labels

    # 5. Phase 4: Customer Segment Profiling & Neutral Naming
    cluster_profiles = {}
    segment_names = {}

    # Rank clusters based on Monetary and Frequency to assign data-derived labels
    cluster_stats = rfm_df.groupby('Cluster').agg(
        Count=('CustomerID', 'count'),
        AvgRecency=('Recency', 'mean'),
        AvgFrequency=('Frequency', 'mean'),
        AvgMonetary=('Monetary', 'mean'),
        MedRecency=('Recency', 'median'),
        MedFrequency=('Frequency', 'median'),
        MedMonetary=('Monetary', 'median')
    )

    # Assign data-derived names based on statistical profiles:
    # Rank clusters by Monetary & Frequency
    sorted_by_value = cluster_stats.sort_values(by=['AvgMonetary', 'AvgFrequency'], ascending=False).index.tolist()
    sorted_by_recency = cluster_stats.sort_values(by=['AvgRecency'], ascending=True).index.tolist()

    # Define segment map based on rank
    for cluster_id in range(best_k):
        stats = cluster_stats.loc[cluster_id]
        c_count = int(stats['Count'])
        c_rec = round(float(stats['AvgRecency']), 1)
        c_freq = round(float(stats['AvgFrequency']), 1)
        c_mon = round(float(stats['AvgMonetary']), 2)

        if cluster_id == sorted_by_value[0]:
            name = "High-Value Customers"
        elif cluster_id == sorted_by_value[-1]:
            name = "Low-Activity Customers"
        elif cluster_id in sorted_by_recency[:2]:
            name = "Recent Customers"
        else:
            name = "Frequent Customers"

        segment_names[int(cluster_id)] = name
        cluster_profiles[int(cluster_id)] = {
            'segment_name': name,
            'customer_count': c_count,
            'percentage': round((c_count / len(rfm_df)) * 100, 2),
            'avg_recency_days': c_rec,
            'avg_frequency_orders': c_freq,
            'avg_monetary_spend': c_mon
        }

    rfm_df['Segment'] = rfm_df['Cluster'].map(segment_names)

    print("\n" + "=" * 60)
    print("PHASE 4: CUSTOMER SEGMENT CHARACTERISTICS")
    print("=" * 60)
    for c_id, profile in cluster_profiles.items():
        print(f"Cluster {c_id}: {profile['segment_name']}")
        print(f"  - Count     : {profile['customer_count']:,} ({profile['percentage']}%)")
        print(f"  - Avg Recency: {profile['avg_recency_days']} days")
        print(f"  - Avg Frequency: {profile['avg_frequency_orders']} orders")
        print(f"  - Avg Monetary: £{profile['avg_monetary_spend']:,}")
        print()

    # 6. Phase 5: Product Recommendations per Cluster
    # Merge cluster IDs back to clean transactions to identify cluster-level top items
    merged_clean = clean_df.merge(rfm_df[['CustomerID', 'Cluster', 'Segment']], on='CustomerID', how='left')

    top_products_per_cluster = {}
    for cluster_id in range(best_k):
        cluster_tx = merged_clean[merged_clean['Cluster'] == cluster_id]

        top_items = (
            cluster_tx.groupby(['StockCode', 'Description'])
            .agg(
                total_quantity=('Quantity', 'sum'),
                total_orders=('InvoiceNo', 'nunique'),
                unique_buyers=('CustomerID', 'nunique'),
                unit_price=('UnitPrice', 'mean')
            )
            .reset_index()
            .sort_values(by=['unique_buyers', 'total_quantity'], ascending=False)
            .head(25)
        )

        top_list = []
        for _, row in top_items.iterrows():
            top_list.append({
                'stock_code': str(row['StockCode']),
                'description': str(row['Description']),
                'total_quantity': int(row['total_quantity']),
                'total_orders': int(row['total_orders']),
                'unique_buyers': int(row['unique_buyers']),
                'unit_price': round(float(row['unit_price']), 2)
            })

        top_products_per_cluster[str(cluster_id)] = top_list

    # 7. Phase 6: Persistence
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'kmeans.pkl'))

    with open(TOP_PRODUCTS_PATH, 'w') as f:
        json.dump(top_products_per_cluster, f, indent=2)

    rfm_df.to_csv(CLUSTERED_RFM_PATH, index=False)

    metadata = {
        'dataset_source': 'Online Retail.xlsx',
        'cleaned_rows': len(clean_df),
        'num_customers': len(rfm_df),
        'features_used': ['Recency', 'Frequency', 'Monetary'],
        'supporting_features': ['TotalQuantity', 'AvgOrderValue', 'UniqueProducts', 'PrimaryCountry'],
        'selected_k': best_k,
        'best_silhouette_score': float(round(best_silhouette, 4)),
        'k_evaluation': evaluation_results,
        'cluster_profiles': cluster_profiles,
        'trained_at': pd.Timestamp.now().isoformat()
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("=" * 60)
    print(f"[PIPELINE COMPLETE] All models & artifacts saved in {MODELS_DIR}")
    print("=" * 60)
    return metadata
