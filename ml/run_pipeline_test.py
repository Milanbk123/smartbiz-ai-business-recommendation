"""
Execution test for RFM + K-Means Pipeline on real Online Retail.xlsx
"""

import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.pipeline import train_pipeline
from ml.recommender import recommend_for_customer, predict_segment_and_recommend

if __name__ == '__main__':
    print("Starting ML Pipeline execution on Online Retail.xlsx...")
    metadata = train_pipeline(force_reprocess=True)

    print("\nTesting Customer Recommendation Lookup...")
    # Sample customer recommendation
    sample_cust_id = 17850
    rec_result = recommend_for_customer(sample_cust_id, top_n=5)
    print(f"\nRecommendation result for Customer {sample_cust_id}:")
    print(json.dumps(rec_result, indent=2))

    print("\nTesting Ad-Hoc RFM Recommendation Input...")
    rfm_result = predict_segment_and_recommend(recency=15, frequency=10, monetary=2500, top_n=5)
    print("\nAd-Hoc RFM Result:")
    print(json.dumps(rfm_result, indent=2))
