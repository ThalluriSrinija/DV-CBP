# clustering.py

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from rfm import create_rfm
from preprocessing import load_and_clean_data


# ---- Step 1: Scale RFM Features ----
def scale_rfm(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    return rfm_scaled


# ---- Step 2: Perform Clustering (FIXED) ----
def perform_clustering(rfm, n_clusters=4):

    # 🔥 IMPORTANT FIX: Adjust clusters based on data size
    n_clusters = min(n_clusters, len(rfm))

    rfm_scaled = scale_rfm(rfm)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    return rfm


# ---- Step 3: Label Clusters (Business Meaning) ----
def label_clusters(rfm):

    if rfm.empty:
        return rfm

    cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

    labels = {}

    for cluster in cluster_summary.index:
        r = cluster_summary.loc[cluster, 'Recency']
        f = cluster_summary.loc[cluster, 'Frequency']
        m = cluster_summary.loc[cluster, 'Monetary']

        # ---- Improved Logic ----
        if m > 5000 and f > 5:
            labels[cluster] = "High Value Customers"
        elif r < 30:
            labels[cluster] = "Recent Customers"
        elif f < 2:
            labels[cluster] = "Low Engagement"
        else:
            labels[cluster] = "Regular Customers"

    rfm['Segment'] = rfm['Cluster'].map(labels)

    return rfm


# ---- Step 4: Elbow Method (FIXED) ----
def calculate_elbow(rfm):

    if len(rfm) < 2:
        return [], []

    rfm_scaled = scale_rfm(rfm)

    inertia = []
    K = []

    # 🔥 IMPORTANT FIX: Dynamic cluster range
    max_k = min(10, len(rfm_scaled))

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        inertia.append(kmeans.inertia_)
        K.append(k)

    return K, inertia


# ---- Test Run ----
if __name__ == "__main__":
    df = load_and_clean_data("data/cleaned_online_retail.xlsx")
    rfm = create_rfm(df)

    rfm = perform_clustering(rfm)
    rfm = label_clusters(rfm)

    print("\nClustered Data:")
    print(rfm.head())

    print("\nCluster Summary:")
    print(rfm.groupby('Segment').mean())