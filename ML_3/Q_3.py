import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Load the dataset
df = pd.read_excel("SMART_dataset.xlsx")

# Select only numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns

# Choose two feature vectors (rows) and fill NaNs with 0
vec1 = df.iloc[0][numeric_cols].fillna(0).values
vec2 = df.iloc[1][numeric_cols].fillna(0).values

# Compute Minkowski distance for r = 1 to 10
r_values = list(range(1, 11))
distances = [distance.minkowski(vec1, vec2, p=r) for r in r_values]

# Print distances
print("Minkowski Distances (r=1 to 10):")
for r, d in zip(r_values, distances):
    print(f"r = {r}: Distance = {d}")

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(r_values, distances, marker='o', linestyle='-')
plt.title("Minkowski Distance between Two Vectors (r = 1 to 10)")
plt.xlabel("r (Order of Minkowski Distance)")
plt.ylabel("Distance")
plt.grid(True)
plt.tight_layout()
plt.show()
