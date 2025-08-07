import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel('SMART_dataset.xlsx')

# Select numerical SMART attributes (drop target like 'failure' or 'smart_9_raw')
feature_cols = ['smart_5_raw', 'smart_187_raw', 'smart_197_raw', 'smart_198_raw']  

# Drop rows with NaN in selected features
X = df[feature_cols].dropna()

# Scale the data (important for KMeans to avoid bias due to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering with k = 2
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
kmeans.fit(X_scaled)

# Output cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

print("\n Cluster Labels (first 10):", labels[:10])
print("\n Cluster Centers (scaled values):\n", centers)
