import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Load Excel file
df = pd.read_excel('DataSet2.xlsx', engine='openpyxl')

# Fill missing values or drop rows with NaNs
df = df.dropna()

# Select any two feature vectors (rows), ensure numeric features only
vec1 = df.iloc[0].values
vec2 = df.iloc[1].values

# Ensure only numeric columns (if dataset has strings)
vec1 = vec1.astype(float)
vec2 = vec2.astype(float)

# Calculate Minkowski distances for r = 1 to 10
r_values = list(range(1, 11))
distances = []

for r in r_values:
    mink_dist = distance.minkowski(vec1, vec2, p=r)
    distances.append(mink_dist)

# Plot the result
plt.plot(r_values, distances, marker='o')
plt.title('Minkowski Distance (r = 1 to 10)')
plt.xlabel('r value')
plt.ylabel('Distance')
plt.grid(True)
plt.show()

# Optional: print the values
for r, d in zip(r_values, distances):
    print(f"r={r}: Distance={d:.4f}")
