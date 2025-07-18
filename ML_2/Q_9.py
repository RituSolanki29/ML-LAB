import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load data
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Work on a copy
df_scaled = df.copy()

# Identify numeric columns only
numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns

# OPTIONAL: Remove columns with constant value (not useful for scaling)
numeric_cols = [col for col in numeric_cols if df_scaled[col].nunique() > 1]

# Fill missing values before scaling (can use your imputed version here)
df_scaled[numeric_cols] = df_scaled[numeric_cols].fillna(df_scaled[numeric_cols].median())

# Choose scaler: MinMax or Standard
scaler = MinMaxScaler()   # You can switch to StandardScaler() if needed

# Apply scaling
df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

# View the scaled data
print(df_scaled.head())
