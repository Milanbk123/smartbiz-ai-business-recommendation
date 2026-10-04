"""
Inference wrapper for SmartBiz AI recommendation system.
Forwards inference calls to ml.recommender.
"""

from .recommender import recommend_for_customer, predict_segment_and_recommend


def predict(customer_id_or_data):
    """
    Wrapper for recommendation inference.
    """
    if isinstance(customer_id_or_data, (int, str)) and str(customer_id_or_data).isdigit():
        return recommend_for_customer(int(customer_id_or_data))
    elif isinstance(customer_id_or_data, dict):
        rec = customer_id_or_data.get('recency', 30)
        freq = customer_id_or_data.get('frequency', 5)
        mon = customer_id_or_data.get('monetary', 1000)
        return predict_segment_and_recommend(rec, freq, mon)
    else:
        raise ValueError("Invalid input for recommendation predictor.")
