# rfm.py

import pandas as pd
from preprocessing import load_and_clean_data

def create_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    return rfm


if __name__ == "__main__":
    df = load_and_clean_data("data/cleaned_online_retail.xlsx")
    rfm = create_rfm(df)

    print(rfm.head())