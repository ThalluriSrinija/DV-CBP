# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import load_and_clean_data
from rfm import create_rfm
from clustering import perform_clustering, label_clusters, calculate_elbow
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# ---- TITLE ----
st.title("E-Commerce Customer Segmentation Dashboard")

st.markdown("""
## 📊 Customer Segmentation Analysis
This dashboard analyzes customer behavior using RFM and K-Means clustering.
---
""")

# ---- LOAD DATA ----
df = load_and_clean_data("data/cleaned_online_retail.xlsx")

# ---- SIDEBAR FILTERS ----
st.sidebar.header("Filters")
st.sidebar.info("Use filters to explore customer behavior")

countries = sorted(df['Country'].dropna().unique())
selected_country = st.sidebar.selectbox("Select Country", ["All"] + countries)

# ---- APPLY COUNTRY FILTER ----
filtered_df = df.copy()
if selected_country != "All":
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]

# ---- RFM AFTER FILTER ----
filtered_rfm = create_rfm(filtered_df)
filtered_rfm = perform_clustering(filtered_rfm)
filtered_rfm = label_clusters(filtered_rfm)

# ---- SEGMENT FILTER ----
segments = sorted(filtered_rfm['Segment'].dropna().unique())
selected_segment = st.sidebar.selectbox("Select Segment", ["All"] + segments)

if selected_segment != "All":
    filtered_rfm = filtered_rfm[filtered_rfm['Segment'] == selected_segment]

segment_counts = filtered_rfm['Segment'].value_counts()

# ---- GRAPH HELPERS ----
def plot_hist(data, column, title):
    fig, ax = plt.subplots(figsize=(8,4))
    data[column].hist(ax=ax, bins=40)
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def plot_scatter(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ---- TABS ----
tab1, tab2 = st.tabs(["📊 Basic Analysis", "📈 Advanced Analysis"])

# =========================
# BASIC TAB
# =========================
with tab1:

    st.subheader("Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Revenue", f"{filtered_df['TotalPrice'].sum():,.2f}")
    col2.metric("Total Customers", filtered_df['CustomerID'].nunique())
    col3.metric("Avg Order Value", f"{filtered_df['TotalPrice'].mean():,.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Segments")
        st.bar_chart(segment_counts)

    with col2:
        st.subheader("Top Countries")
        country_sales = filtered_df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
        st.bar_chart(country_sales.head(10))

    st.markdown("---")

    st.subheader("Customer Data")
    st.dataframe(filtered_rfm.head(50))

# =========================
# ADVANCED TAB
# =========================
with tab2:

    st.subheader("Advanced Analysis")

    # 1 Revenue Distribution
    plot_hist(filtered_df, 'TotalPrice', "Revenue Distribution")

    # 2 Orders Over Time
    st.subheader("Orders Over Time")
    orders_time = filtered_df.groupby(filtered_df['InvoiceDate'].dt.date).size()
    st.line_chart(orders_time)

    # 3 Revenue Over Time
    st.subheader("Revenue Over Time")
    revenue_time = filtered_df.groupby(filtered_df['InvoiceDate'].dt.date)['TotalPrice'].sum()
    st.line_chart(revenue_time)

    # 4 Top Products
    st.subheader("Top Products")
    top_products = filtered_df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)

    # 5–8 Distributions
    plot_hist(filtered_df, 'Quantity', "Quantity Distribution")
    plot_hist(filtered_rfm, 'Recency', "Recency Distribution")
    plot_hist(filtered_rfm, 'Frequency', "Frequency Distribution")
    plot_hist(filtered_rfm, 'Monetary', "Monetary Distribution")

    # 9 Pie Chart (FIXED)
    st.subheader("Segment Distribution")

    fig, ax = plt.subplots(figsize=(6,6))
    segment_counts.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
    ax.set_ylabel("")
    ax.axis('equal')  # 🔥 IMPORTANT FIX
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    # 10 Avg Spend
    st.subheader("Avg Spend per Segment")
    avg_spend = filtered_rfm.groupby('Segment')['Monetary'].mean()
    st.bar_chart(avg_spend)

    # 11–12 Scatter
    plot_scatter(filtered_rfm['Frequency'], filtered_rfm['Monetary'],
                 "Frequency", "Monetary", "Frequency vs Monetary")

    plot_scatter(filtered_rfm['Recency'], filtered_rfm['Frequency'],
                 "Recency", "Frequency", "Recency vs Frequency")

    # 13 Customers by Country
    st.subheader("Customers by Country")
    country_customers = filtered_df.groupby('Country')['CustomerID'].nunique()
    st.bar_chart(country_customers.head(10))

    # 14 Monthly Revenue
    st.subheader("Monthly Revenue")
    temp_df = filtered_df.copy()
    temp_df['Month'] = temp_df['InvoiceDate'].dt.to_period('M')
    monthly_revenue = temp_df.groupby('Month')['TotalPrice'].sum()
    st.line_chart(monthly_revenue)

    # Elbow Method (SAFE)
    st.subheader("Elbow Method")

    if len(filtered_rfm) > 5:
        K, inertia = calculate_elbow(filtered_rfm)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(K, inertia, marker='o')
        ax.set_title("Elbow Method")
        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("Not enough data for clustering")