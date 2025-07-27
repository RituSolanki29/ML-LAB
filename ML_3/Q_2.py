import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("SMART_dataset.xlsx")

# Select the feature to analyze (replace with any other feature if needed)
feature = 'smart_9_raw'

# Drop NaN values for this feature
feature_data = df[feature].dropna()

# Compute histogram data
hist_values, bin_edges = np.histogram(feature_data, bins=10)

# Calculate mean and variance
mean = np.mean(feature_data)
variance = np.var(feature_data)

print(f"Feature: {feature}")
print(f"Mean: {mean}")
print(f"Variance: {variance}")

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(feature_data, bins=10, color='skyblue', edgecolor='black')
plt.title(f"Histogram of {feature}")
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
