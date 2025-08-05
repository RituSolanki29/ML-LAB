import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Excel file
df = pd.read_excel('DataSet2.xlsx', engine='openpyxl')

feature_name = 'smart_1_normalized' 

# Drop NaN values for this feature
feature_data = df[feature_name].dropna()

# 1. Plot histogram with buckets
plt.hist(feature_data, bins=10, edgecolor='black')  
plt.title(f'Histogram of {feature_name}')
plt.xlabel(feature_name)
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 2. Calculate mean and variance
mean_value = np.mean(feature_data)
variance_value = np.var(feature_data)

print(f"\nFeature: {feature_name}")
print(f"Mean: {mean_value:.4f}")
print(f"Variance: {variance_value:.4f}")
