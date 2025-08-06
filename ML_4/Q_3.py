import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Generate 20 random data points for features X and Y
np.random.seed(0)  
X = np.random.uniform(1, 10, 20)
Y = np.random.uniform(1, 10, 20)

# Step 2: Create class labels based on a simple rule
# If X + Y > 10 → class 1 (Red), else class 0 (Blue)
classes = np.where(X + Y > 10, 1, 0)

# Step 3: Create a DataFrame (optional, for easy display)
df = pd.DataFrame({'X': X, 'Y': Y, 'Class': classes})

# Step 4: Plot the scatter plot
plt.figure(figsize=(8, 6))

# Plot class 0 (Blue)
plt.scatter(df[df['Class'] == 0]['X'], df[df['Class'] == 0]['Y'], color='blue', label='Class 0 (Blue)', s=80)

# Plot class 1 (Red)
plt.scatter(df[df['Class'] == 1]['X'], df[df['Class'] == 1]['Y'], color='red', label='Class 1 (Red)', s=80)

plt.title('Scatter Plot of 20 Random Data Points with Classes')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.legend()
plt.grid(True)
plt.show()
