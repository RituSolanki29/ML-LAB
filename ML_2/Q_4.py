import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the Excel sheet
df = pd.read_excel('Lab Session Data.xlsx', sheet_name="thyroid0387_UCI", engine='openpyxl')

print("Preview of the Dataset:")
print(df.head())

# Step 2: Identify data types
print("\nData Types of Each Attribute:")
print(df.dtypes)

# Step 3: Detect categorical attributes
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

print("\nCategorical Attributes:")
print(categorical_cols)

print("\nNumeric Attributes:")
print(numeric_cols)

# Step 4: Encoding Scheme Suggestion
print("\nEncoding Scheme Suggestion:")
for col in categorical_cols:
    unique_vals = df[col].unique()
    print(f"{col} → Unique values: {unique_vals}")
    if len(unique_vals) <= 10:
        print("  Suggested: One-Hot Encoding (Nominal)\n")
    else:
        print("  Suggested: Label Encoding (Likely Ordinal or High Cardinality)\n")

# Step 5: Range of numeric attributes
print("\nRange of Numeric Attributes:")
for col in numeric_cols:
    print(f"{col}: min = {df[col].min()}, max = {df[col].max()}")

# Step 6: Missing Values
print("\nMissing Values in Each Attribute:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

# Step 7: Outlier Detection (IQR method)
print("\nOutliers in Numeric Attributes (using IQR):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

# Step 8: Mean and Standard Deviation
print("\nMean and Standard Deviation of Numeric Attributes:")
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    print(f"{col}: Mean = {mean:.2f}, Std Dev = {std:.2f}")
