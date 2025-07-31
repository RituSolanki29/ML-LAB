import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Step 1: Load your project data
df = pd.read_excel('DataSet2.xlsx', engine='openpyxl')  

# Step 2: Select 2 features and the target class (replace with your own)
feature_x = 'smart_5_raw'    
feature_y = 'smart_187_raw'  
target = 'failure'          
# Drop rows with missing values in selected columns
df = df[[feature_x, feature_y, target]].dropna()

# For training, randomly select 20 samples (like A3)
train_data = df.sample(n=20, random_state=42)
X_train = train_data[[feature_x, feature_y]].values
y_train = train_data[target].values

# A3: Scatter plot of training data
plt.figure(figsize=(6, 5))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Class 1')
plt.title('A3: Training Data')
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.legend()
plt.grid(True)
plt.show()

# A4 and A5: Classify dense test grid with different k values
k_values = [1, 3, 5, 7]

# Create mesh grid (X & Y from 0 to 10, increment 0.1)
x_min, x_max = 0, 10
y_min, y_max = 0, 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
grid_points = np.c_[xx.ravel(), yy.ravel()]  # Shape (10000, 2)

for k in k_values:
    # Train kNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on the grid
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Class 0 (Train)')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Class 1 (Train)')
    plt.title(f'A4/A5: k = {k} Decision Boundary')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt.grid(True)
    plt.show()
