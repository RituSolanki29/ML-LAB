import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data 
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Take first 20 observation vectors
df_20 = df.head(20)

# Assuming binary attributes 
binary_cols = [col for col in df_20.columns 
               if set(df_20[col].dropna().unique()).issubset({0, 1})]

binary_df = df_20[binary_cols]
# compute JC and SMC
def jaccard_coefficient(x, y):
    f11 = np.sum((x == 1) & (y == 1))
    f10 = np.sum((x == 1) & (y == 0))
    f01 = np.sum((x == 0) & (y == 1))
    return f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0

def simple_matching_coefficient(x, y):
    f11 = np.sum((x == 1) & (y == 1))
    f00 = np.sum((x == 0) & (y == 0))
    f10 = np.sum((x == 1) & (y == 0))
    f01 = np.sum((x == 0) & (y == 1))
    return (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0

#Compute JC and SMC Matrices
jc_matrix = np.zeros((20, 20))
smc_matrix = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        jc_matrix[i, j] = jaccard_coefficient(binary_df.iloc[i], binary_df.iloc[j])
        smc_matrix[i, j] = simple_matching_coefficient(binary_df.iloc[i], binary_df.iloc[j])

# Convert all to numeric
numeric_df_20 = df_20.select_dtypes(include=[np.number]).fillna(0)
cos_matrix = cosine_similarity(numeric_df_20)

#Plot Heatmaps
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(jc_matrix, annot=False, cmap="YlGnBu")
plt.title("Jaccard Coefficient Heatmap")

plt.subplot(1, 3, 2)
sns.heatmap(smc_matrix, annot=False, cmap="YlOrBr")
plt.title("Simple Matching Coefficient Heatmap")

plt.subplot(1, 3, 3)
sns.heatmap(cos_matrix, annot=False, cmap="viridis")
plt.title("Cosine Similarity Heatmap")

plt.tight_layout()
plt.show()
