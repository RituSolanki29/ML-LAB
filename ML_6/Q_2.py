import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel('DataSet2.xlsx')

def calculate_gini(series):
 
    value_counts = series.value_counts(normalize=True)
    probabilities = value_counts.values
    gini = 1 - np.sum(probabilities ** 2)
    return gini

gini_value = calculate_gini(df['failure'])
print("Gini Index:", gini_value)
