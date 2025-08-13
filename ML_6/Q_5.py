import pandas as pd
import numpy as np
from collections import Counter

# -----------------------------
# A4: Binning Function
# -----------------------------
def bin_feature(series, num_bins=3, binning_type="equal_width"):
    """
    Convert continuous features to categorical bins.
    Parameters:
        series: pandas Series (continuous values)
        num_bins: int, number of bins
        binning_type: "equal_width" or "equal_freq"
    Returns:
        binned categorical series
    """
    if binning_type == "equal_width":
        return pd.cut(series, bins=num_bins, labels=False)
    elif binning_type == "equal_freq":
        return pd.qcut(series, q=num_bins, labels=False, duplicates='drop')
    else:
        raise ValueError("Invalid binning type! Use 'equal_width' or 'equal_freq'.")

# -----------------------------
# Entropy & Information Gain
# -----------------------------
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def information_gain(y, x):
    """ Calculate Information Gain of a feature x for target y """
    base_entropy = entropy(y)
    values, counts = np.unique(x, return_counts=True)
    weighted_entropy = sum((counts[i] / len(x)) * entropy(y[x == values[i]]) for i in range(len(values)))
    return base_entropy - weighted_entropy

# -----------------------------
# Detect Best Feature (Root node)
# -----------------------------
def best_feature_to_split(X, y):
    """ Return the feature with the highest information gain """
    gains = {}
    for col in X.columns:
        gain = information_gain(y, X[col])
        gains[col] = gain
    best = max(gains, key=gains.get)
    return best, gains

# -----------------------------
# Decision Tree Class
# -----------------------------
class DecisionTree:
    def __init__(self, max_depth=5, binning_type="equal_width", num_bins=3):
        self.max_depth = max_depth
        self.binning_type = binning_type
        self.num_bins = num_bins
        self.tree = None

    def fit(self, X, y, depth=0):
        # Convert continuous columns to categorical using binning
        X_binned = X.copy()
        for col in X_binned.columns:
            if np.issubdtype(X_binned[col].dtype, np.number):
                X_binned[col] = bin_feature(X_binned[col], self.num_bins, self.binning_type)

        # Stopping conditions
        if len(set(y)) == 1:  # Pure node
            return y.iloc[0]
        if depth >= self.max_depth or X_binned.empty:  # Max depth or no features
            return Counter(y).most_common(1)[0][0]

        # Find best feature
        best_feature, _ = best_feature_to_split(X_binned, y)

        # Create tree dict
        tree = {best_feature: {}}
        for value in np.unique(X_binned[best_feature]):
            sub_X = X_binned[X_binned[best_feature] == value].drop(columns=[best_feature])
            sub_y = y[X_binned[best_feature] == value]
            subtree = self.fit(sub_X, sub_y, depth + 1)
            tree[best_feature][value] = subtree

        self.tree = tree
        return tree

    def predict_one(self, x, tree=None):
        if tree is None:
            tree = self.tree
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        value = x.get(feature)
        if value in tree[feature]:
            return self.predict_one(x, tree[feature][value])
        else:
            return Counter([subtree for subtree in tree[feature].values() if not isinstance(subtree, dict)]).most_common(1)[0][0]

    def predict(self, X):
        return X.apply(lambda row: self.predict_one(row.to_dict()), axis=1)


if __name__ == "__main__":
    df = pd.read_excel("DataSet2.xlsx")
   
    X = df.drop(columns=["failure"])
    y = df["failure"]

    tree = DecisionTree(max_depth=3, binning_type="equal_freq", num_bins=3)
    trained_tree = tree.fit(X, y)
    print("Decision Tree:", trained_tree)

    preds = tree.predict(X)
    print("Predictions:", preds.values)

