import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Load your dataset 
df = pd.read_excel('DataSet2.xlsx')  

# Pick two actual feature columns (normalized values)
feature1 = 'smart_5_normalized'
feature2 = 'smart_9_normalized'

# Drop rows with missing values in selected features
df_clean = df[[feature1, feature2]].dropna()

# Ensure at least two valid rows to compare
if len(df_clean) < 2:
    raise ValueError("❌ Not enough rows with valid values for the selected features.")

# Select two rows (as vectors)
vec1 = df_clean.iloc[0].values
vec2 = df_clean.iloc[1].values

# Calculate Minkowski distances for r = 1 to 10
r_values = list(range(1, 11))
distances = [distance.minkowski(vec1, vec2, p=r) for r in r_values]

# Plot
plt.plot(r_values, distances, marker='o')
plt.title("Minkowski Distance (r = 1 to 10)")
plt.xlabel("r (Order)")
plt.ylabel("Distance")
plt.grid(True)
plt.show()
