# data/export_from_bq.py
# Pulls rfm_scores from BigQuery and saves to data/features.csv
# Run this once before training: python data/export_from_bq.py

from google.cloud import bigquery
import pandas as pd
import os

PROJECT = "customer-churn-492703"
QUERY   = """
    SELECT
        user_id,
        segment,
        traffic_source,
        country,
        age,
        recency_days,
        frequency,
        monetary,
        avg_order_value,
        active_months,
        r_score,
        f_score,
        m_score,
        churned
    FROM `customer-churn-492703.customer_intelligence.rfm_scores`
"""

client = bigquery.Client(project=PROJECT)
print("Pulling data from BigQuery...")
df = client.query(QUERY).to_dataframe()
print(f"Retrieved {len(df):,} rows with columns: {list(df.columns)}")

out_path = os.path.join(os.path.dirname(__file__), "features.csv")
df.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
print(f"\nChurn split:")
print(df["churned"].value_counts(normalize=True).round(3))