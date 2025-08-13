import pandas as pd
import numpy as np
from math import log2

# Function to calculate entropy
def entropy(data, target_col):
    values, counts = np.unique(data[target_col], return_counts=True)
    entropy_val = 0
    for i in range(len(values)):
        prob = counts[i] / np.sum(counts)
        entropy_val -= prob * log2(prob)
    return entropy_val

# Function to calculate information gain
def information_gain(data, feature, target_col):
    total_entropy = entropy(data, target_col)
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = 0
    for i in range(len(values)):
        subset = data[data[feature] == values[i]]
        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(subset, target_col)
    return total_entropy - weighted_entropy

# Equal width binning with fix for constant columns
def equal_width_binning(series, bins=4):
    if series.nunique() == 1:  # All values same
        return pd.Series([0] * len(series), index=series.index)  # Single bin
    return pd.cut(series, bins=bins, labels=False, include_lowest=True, duplicates="drop")

# Find root node
def find_root_node(data, target_col):
    ig_dict = {}
    temp_data = data.copy()
    
    for feature in data.columns:
        if feature != target_col:
            # If numeric, bin into categories
            if pd.api.types.is_numeric_dtype(data[feature]):
                temp_data[feature] = equal_width_binning(temp_data[feature], bins=4)
            ig = information_gain(temp_data, feature, target_col)
            ig_dict[feature] = ig
    
    root = max(ig_dict, key=ig_dict.get)
    return root, ig_dict


# ---------- Step 5: Apply on SMART dataset ----------
df = pd.read_excel("DataSet2.xlsx")
target_col = "failure"

root, ig_dict = find_root_node(df, target_col)

print(f"Root Node Feature: {root}")
print("\nInformation Gain for each feature:")
for feat, ig in ig_dict.items():
    print(f"{feat}: {ig:.4f}")
