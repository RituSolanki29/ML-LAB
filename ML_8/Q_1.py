import pandas as pd
import numpy as np  # <-- Missing import
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel('DataSet2.xlsx')

target_column = 'failure'

def calculate_entropy(series):

    # Probability of each category
    value_counts = series.value_counts(normalize=True)
    probabilities = value_counts.values

    # Avoid log(0)
    probabilities = probabilities[probabilities > 0]

    # Entropy formula: -Σ (p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

entropy_value = calculate_entropy(df[target_column])
print("Entropy:", entropy_value)
