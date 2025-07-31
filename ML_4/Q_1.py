import pandas as pd
import numpy as np

# Load Excel file
df = pd.read_excel('DataSet2.xlsx', engine='openpyxl')

# Get SMART normalized columns
smart_normalized_columns = [col for col in df.columns if 'normalized' in col and col.startswith('smart_')]

# Convert to numeric (important for Excel imports)
for col in smart_normalized_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check class counts
print("Class 0 count (failure=0):", len(df[df['failure'] == 0]))
print("Class 1 count (failure=1):", len(df[df['failure'] == 1]))

# Drop rows with any NaN in SMART columns
class0_data = df[df['failure'] == 0][smart_normalized_columns].dropna()
class1_data = df[df['failure'] == 1][smart_normalized_columns].dropna()

# Check if there is any data left
if class0_data.empty or class1_data.empty:
    print("One or both classes have no valid rows after dropping NaNs.")
else:
    # Compute centroid and spread
    centroid0 = class0_data.mean(axis=0)
    centroid1 = class1_data.mean(axis=0)

    spread0 = class0_data.std(axis=0)
    spread1 = class1_data.std(axis=0)

    interclass_distance = np.linalg.norm(centroid0 - centroid1)

    # Print results
    print("\nCentroid (Mean) of Class 0 (No Failure):\n", centroid0)
    print("\nSpread of Class 0 (No Failure):\n", spread0)
    print("\nCentroid (Mean) of Class 1 (Failure):\n", centroid1)
    print("\nSpread of Class 1 (Failure):\n", spread1)
    print(f"\nInterclass Euclidean Distance between Class 0 and Class 1: {interclass_distance:.4f}")
