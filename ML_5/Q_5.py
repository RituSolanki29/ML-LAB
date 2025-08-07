import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load dataset
df = pd.read_excel('SMART_dataset.xlsx')

# Choose numerical features only (exclude target column like 'failure')
feature_cols = ['smart_5_raw', 'smart_187_raw', 'smart_197_raw', 'smart_198_raw']
X = df[feature_cols].dropna()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans with k=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X_scaled)

# Get cluster labels
labels = kmeans.labels_

# Compute clustering metrics
sil_score = silhouette_score(X_scaled, labels)
ch_score = calinski_harabasz_score(X_scaled, labels)
db_index = davies_bouldin_score(X_scaled, labels)

# Print the results
print(f"Silhouette Score        : {sil_score:.4f}  (Higher is better, max = 1)")
print(f"Calinski-Harabasz Score : {ch_score:.4f}  (Higher is better)")
print(f"Davies-Bouldin Index    : {db_index:.4f}  (Lower is better)")
