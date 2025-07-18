import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Copy to avoid changes to original
df_imputed = df.copy()

# Separate numeric and categorical columns
numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns

# Helper function to detect outliers using IQR
def has_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).any()

# Impute numeric columns
for col in numeric_cols:
    if df_imputed[col].isnull().sum() > 0:
        if has_outliers(df_imputed[col].dropna()):
            # Use median if outliers are present
            median_val = df_imputed[col].median()
            df_imputed[col].fillna(median_val, inplace=True)
        else:
            # Use mean if no outliers
            mean_val = df_imputed[col].mean()
            df_imputed[col].fillna(mean_val, inplace=True)

# Impute categorical columns with mode
for col in categorical_cols:
    if df_imputed[col].isnull().sum() > 0:
        mode_val = df_imputed[col].mode()[0]
        df_imputed[col].fillna(mode_val, inplace=True)
