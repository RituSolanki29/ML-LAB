import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Replace '?' with NaN and handle missing data
df.replace('?', pd.NA, inplace=True)

# Convert all values to appropriate data types
for col in df.columns:
    if df[col].dtype == object:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            df[col] = df[col].astype(str)  # force as string for encoding

# Fill missing values: use mode for strings, median for numbers
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Label Encoding for categorical (object) columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == object:
        df[col] = le.fit_transform(df[col].astype(str))

# Extract vectors of the first two observations
vec1 = df.iloc[0].values.reshape(1, -1)
vec2 = df.iloc[1].values.reshape(1, -1)

# Compute cosine similarity
similarity = cosine_similarity(vec1, vec2)

print("Cosine Similarity between observation 1 and 2:", similarity[0][0])
