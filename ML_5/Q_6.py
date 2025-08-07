import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel('SMART_dataset.xlsx')

# Select numerical SMART attributes (replace as needed)
feature_cols = ['smart_5_raw', 'smart_187_raw', 'smart_197_raw', 'smart_198_raw']
X = df[feature_cols].dropna()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize lists to store metric values
k_values = range(2, 11)  # Test k from 2 to 10
sil_scores = []
ch_scores = []
db_scores = []

# Loop through each k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    sil_scores.append(silhouette_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# Plot all three metrics
plt.figure(figsize=(16, 4))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(k_values, sil_scores, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")

# Calinski-Harabasz Score
plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o')
plt.title("Calinski-Harabasz Score vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("CH Score")

# Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(k_values, db_scores, marker='o')
plt.title("Davies-Bouldin Index vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("DB Index")

plt.tight_layout()
plt.show()
