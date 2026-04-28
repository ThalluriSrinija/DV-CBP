# preprocessing.py

import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    # Drop missing CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Remove invalid quantities
    df = df[df['Quantity'] > 0]

    # Convert date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Create TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    return df


if __name__ == "__main__":
    df = load_and_clean_data("cleaned_online_retail.xlsx")
    print(df.head())