"""
Data Processor Module for Online Retail Dataset
Handles loading, cleaning, and RFM feature engineering.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH_1 = os.path.join(BASE_DIR, 'dataset', 'online+retail', 'Online Retail.xlsx')
DATASET_PATH_2 = os.path.join(BASE_DIR, 'dataset', 'Online Retail', 'Online Retail.xlsx')
PROCESSED_TRANSACTIONS_PATH = os.path.join(BASE_DIR, 'dataset', 'cleaned_transactions.csv')
CUSTOMER_RFM_PATH = os.path.join(BASE_DIR, 'dataset', 'customer_rfm.csv')


def get_dataset_path():
    if os.path.exists(DATASET_PATH_1):
        return DATASET_PATH_1
    elif os.path.exists(DATASET_PATH_2):
        return DATASET_PATH_2
    else:
        raise FileNotFoundError(
            f"Dataset not found at either:\n  1. {DATASET_PATH_1}\n  2. {DATASET_PATH_2}"
        )


def load_and_clean_data(force_reprocess=False):
    """
    Phase 1: Load and apply data cleaning rules on Online Retail.xlsx
    Rules applied:
    1. Drop missing CustomerID
    2. Drop cancelled invoices (InvoiceNo starting with 'C')
    3. Drop Quantity <= 0
    4. Drop UnitPrice <= 0
    5. Drop missing Description
    6. Drop duplicate rows
    """
    if not force_reprocess and os.path.exists(PROCESSED_TRANSACTIONS_PATH):
        print(f"[DATA PROCESSOR] Loading cached cleaned transactions from {PROCESSED_TRANSACTIONS_PATH}...")
        df = pd.read_csv(PROCESSED_TRANSACTIONS_PATH, parse_dates=['InvoiceDate'])
        return df

    raw_path = get_dataset_path()
    print(f"[DATA PROCESSOR] Reading Excel file from: {raw_path}...")
    raw_df = pd.read_excel(raw_path, engine='openpyxl')
    initial_rows = len(raw_df)
    print(f"[DATA PROCESSOR] Initial raw row count: {initial_rows:,}")

    # 1. Drop missing CustomerID
    df = raw_df.dropna(subset=['CustomerID']).copy()
    df['CustomerID'] = df['CustomerID'].astype(int)

    # 2. Drop cancelled invoices (InvoiceNo starts with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.upper().str.startswith('C')]

    # 3. Drop Quantity <= 0
    df = df[df['Quantity'] > 0]

    # 4. Drop UnitPrice <= 0
    df = df[df['UnitPrice'] > 0]

    # 5. Drop missing Description
    df = df.dropna(subset=['Description'])
    df['Description'] = df['Description'].astype(str).str.strip()

    # 6. Drop duplicates
    df = df.drop_duplicates()

    # Derived TotalAmount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    cleaned_rows = len(df)
    print(f"[DATA PROCESSOR] Cleaned row count: {cleaned_rows:,} (dropped {initial_rows - cleaned_rows:,} rows)")

    # Save cleaned transactions to csv for fast loading in subsequent runs
    df.to_csv(PROCESSED_TRANSACTIONS_PATH, index=False)
    print(f"[DATA PROCESSOR] Saved cleaned transactions to: {PROCESSED_TRANSACTIONS_PATH}")
    return df


def compute_rfm_features(clean_df, force_reprocess=False):
    """
    Phase 2: Calculate Customer-Level RFM + Supporting Features.
    Features engineered:
    - Recency: Days since customer's last purchase relative to snapshot date
    - Frequency: Count of unique invoices (orders)
    - Monetary: Total spending sum(TotalAmount)
    - TotalQuantity: Sum of quantity purchased
    - AvgOrderValue: Monetary / Frequency
    - UniqueProducts: Number of distinct StockCodes purchased
    - PrimaryCountry: Most frequent country of customer
    """
    if not force_reprocess and os.path.exists(CUSTOMER_RFM_PATH):
        print(f"[DATA PROCESSOR] Loading cached customer RFM from {CUSTOMER_RFM_PATH}...")
        rfm_df = pd.read_csv(CUSTOMER_RFM_PATH)
        return rfm_df

    print("[DATA PROCESSOR] Computing RFM features per customer...")
    snapshot_date = clean_df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm_df = clean_df.groupby('CustomerID').agg(
        LastPurchaseDate=('InvoiceDate', 'max'),
        Recency=('InvoiceDate', lambda dates: (snapshot_date - dates.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalAmount', 'sum'),
        TotalQuantity=('Quantity', 'sum'),
        UniqueProducts=('StockCode', 'nunique'),
        PrimaryCountry=('Country', lambda c: c.mode()[0] if not c.empty else 'Unknown')
    ).reset_index()

    rfm_df['AvgOrderValue'] = (rfm_df['Monetary'] / rfm_df['Frequency']).round(2)
    rfm_df['Monetary'] = rfm_df['Monetary'].round(2)

    print(f"[DATA PROCESSOR] Engineered features for {len(rfm_df):,} unique customers.")
    rfm_df.to_csv(CUSTOMER_RFM_PATH, index=False)
    print(f"[DATA PROCESSOR] Saved customer RFM dataset to: {CUSTOMER_RFM_PATH}")
    return rfm_df
