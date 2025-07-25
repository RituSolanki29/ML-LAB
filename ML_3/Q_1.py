import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

# Load dataset
df = pd.read_excel('SMART_dataset.xlsx')

# Drop non-numeric columns (like date, serial_number, model, etc.)
numeric_df = df.select_dtypes(include=[np.number])

# Separate features and labels
features = numeric_df.drop(columns=['failure'])
labels = numeric_df['failure']

# Check unique classes
print("Class distribution:\n", labels.value_counts())

# Group data by class
class_groups = features.groupby(labels)

# Store centroids and spreads
centroids = {}
spreads = {}

# For each class, calculate centroid and spread
for class_label, group in class_groups:
    centroids[class_label] = group.mean(axis=0)
    spreads[class_label] = group.std(axis=0)

    print(f"\nCentroid for class {class_label}:")
    print(centroids[class_label])

    print(f"\nSpread (standard deviation) for class {class_label}:")
    print(spreads[class_label])

# If at least 2 classes exist, calculate interclass distance
if len(centroids) >= 2:
    class_labels = list(centroids.keys())
    for i in range(len(class_labels)):
        for j in range(i+1, len(class_labels)):
            c1 = centroids[class_labels[i]]
            c2 = centroids[class_labels[j]]
            distance = np.linalg.norm(c1 - c2)
            print(f"\nDistance between class {class_labels[i]} and class {class_labels[j]} centroids: {distance}")
else:
    print("\nOnly one class found. Interclass distance cannot be computed.")