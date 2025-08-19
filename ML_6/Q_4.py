import pandas as pd
import numpy as np

# Function for binning
def binning(column, bins=4, method='width'):
    
    if method == 'width':
        # Equal width binning
        return pd.cut(column, bins=bins, labels=[f"Bin{i+1}" for i in range(bins)])
    elif method == 'frequency':
        # Equal frequency binning
        return pd.qcut(column, q=bins, labels=[f"Bin{i+1}" for i in range(bins)])
    else:
        raise ValueError("Method should be either 'width' or 'frequency'")

# entropy function
def entropy(y):
    probs = y.value_counts(normalize=True)
    return -sum(probs * np.log2(probs))

# Information gain function
def information_gain(df, feature, target):
    total_entropy = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0
    for v in values:
        subset = df[df[feature] == v]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target])
    return total_entropy - weighted_entropy

# Root node detection
def find_root_node(df, target, binning_method='width', bins=4):
    best_feature = None
    best_ig = -1
    
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 2:
            df[col] = binning(df[col], bins=bins, method=binning_method)
        
        ig = information_gain(df, col, target)
        print(f"Feature: {col}, IG: {ig:.4f}")
        
        if ig > best_ig:
            best_ig = ig
            best_feature = col
    
    return best_feature, best_ig

# Sample dataset
df = pd.read_excel("DataSet2.xlsx")
target_col = "failure"

root_feature, ig_value = find_root_node(df, target='failure', binning_method='width', bins=4)
print(f"\nRoot Node: {root_feature}, IG: {ig_value:.4f}")
