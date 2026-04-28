# eda.py

from preprocessing import load_and_clean_data

def perform_eda(df):
    print("\n--- BASIC INFO ---")
    print(df.info())

    print("\n--- TOP COUNTRIES ---")
    print(df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head())

    print("\n--- TOP PRODUCTS ---")
    print(df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head())


if __name__ == "__main__":
    df = load_and_clean_data("data/cleaned_online_retail.xlsx")
    perform_eda(df)