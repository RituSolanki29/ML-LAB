import numpy as np
import pandas as pd

# Manually create the DataFrame since it's small
data = {
    'Customer': ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7', 'C_8', 'C_9', 'C_10'],
    'Candies (#)': [20, 16, 27, 19, 24, 22, 15, 18, 21, 16],
    'Mangoes (Kg)': [6, 3, 6, 1, 4, 1, 4, 4, 1, 2],
    'Milk Packets (#)': [2, 6, 2, 2, 2, 5, 2, 2, 4, 4],
    'Payment (Rs)': [386, 289, 393, 110, 280, 167, 271, 274, 148, 198]
}

df = pd.DataFrame(data)

# Extract A (quantities) and C (total payments)
A = df.iloc[:, 1:-1].values  # Candies, Mangoes, Milk
C = df.iloc[:, -1].values    # Payments

# Compute dimensionality and number of vectors
print(f"Dimensionality of vector space: {A.shape[1]}")
print(f"Number of vectors in vector space: {A.shape[0]}")

# Rank of A
rank = np.linalg.matrix_rank(A)
print(f"Rank of matrix A: {rank}")

# Compute X using pseudo-inverse
X = np.linalg.pinv(A) @ C

# Display costs
product_names = df.columns[1:-1]
print("\nCost of each product:")
for name, price in zip(product_names, X):
    print(f"{name}: ₹{round(price, 2)}")
