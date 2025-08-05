import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Load Excel file
df = pd.read_excel('DataSet2.xlsx', engine='openpyxl')

# Only keep numeric columns
df = df.select_dtypes(include=[np.number])

# Drop rows with NaN only if necessary (keep most rows)
df = df.dropna()


# If there are fewer than 2 rows, raise an error
if df.shape[0] < 2:
    raise ValueError("Not enough rows in the dataset after cleaning to compare vectors.")

# Select two rows (feature vectors)
vec1 = df.iloc[0].values
vec2 = df.iloc[1].values

# Calculate Minkowski distances from r = 1 to 10
r_values = list(range(1, 11))
distances = []

for r in r_values:
    mink_dist = distance.minkowski(vec1, vec2, p=r)
    distances.append(mink_dist)

# Plot the distances
plt.plot(r_values, distances, marker='o')
plt.title('Minkowski Distance (r = 1 to 10)')
plt.xlabel('r value')
plt.ylabel('Distance')
plt.grid(True)
plt.show()
