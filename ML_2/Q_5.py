import pandas as pd

#Load dataset
df = pd.read_excel('Lab Session Data.xlsx', sheet_name="thyroid0387_UCI", engine='openpyxl')

#two vectors
vec1 = df.iloc[0]
vec2 = df.iloc[1]

# Binary valued attributes
binary_columns = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0, 1})]

# Extract binary parts of the vectors
bin_vec1 = vec1[binary_columns].values
bin_vec2 = vec2[binary_columns].values

# Calculate counts needed for JC and SMC
f11 = ((bin_vec1 == 1) & (bin_vec2 == 1)).sum()
f00 = ((bin_vec1 == 0) & (bin_vec2 == 0)).sum()
f10 = ((bin_vec1 == 1) & (bin_vec2 == 0)).sum()
f01 = ((bin_vec1 == 0) & (bin_vec2 == 1)).sum()

# Jaccard Coefficient
JC = f11 / (f01 + f10 + f11)

# Simple Matching Coefficient
SMC = (f11 + f00) / (f00 + f01 + f10 + f11)

print("Jaccard Coefficient (JC):", JC)
print("Simple Matching Coefficient (SMC):", SMC)