import numpy as np 
import pandas as pd 

# Loading Purchase data
f=pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

# Segregating matrix a and c
A = f[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = f['Payment (Rs)'].values

# dimensionality of vector space
print("Matrix A:\n", A)
print("\n Dimensionality of A:", A.shape[1])
print("\n Number of vectors in A",A.shape[0])

#Finding rank
rankA = np.linalg.matrix_rank(A)
print("\n Rank of Matrix A: ", rankA)

# Compute pseudo-inverse
A_pinv = np.linalg.pinv(A)

# Estimate cost per product
X = A_pinv @ C  # X is a matrix of costs

# Display results
product_names = ['Candies ', 'Mangoes ', 'Milk Packets ']
print("\nEstimated cost of each product:")
for name, cost in zip(product_names, X.flatten()):
    print(f"{name}: ₹{cost:.2f}")