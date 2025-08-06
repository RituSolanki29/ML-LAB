import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Generate Training Data (same as before)
np.random.seed(0)
X_train = np.random.uniform(1, 10, 20)
Y_train = np.random.uniform(1, 10, 20)
train_data = np.column_stack((X_train, Y_train))

# Class rule: Class 1 (Red) if X + Y > 10, else Class 0 (Blue)
train_labels = np.where(X_train + Y_train > 10, 1, 0)

# Step 2: Generate Test Set (100x100 grid = 10,000 points)
x_test_vals = np.arange(0, 10.1, 0.1)
y_test_vals = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_test_vals, y_test_vals)
test_data = np.c_[xx.ravel(), yy.ravel()]  # Flattened into 10,000 rows of (X, Y)

# Step 3: Train kNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data, train_labels)

# Step 4: Predict test data classes
test_preds = knn.predict(test_data)

# Step 5: Plot decision boundary (test set with predicted labels)
plt.figure(figsize=(10, 8))

# Plot predicted class 0 (Blue)
plt.scatter(test_data[test_preds == 0, 0], test_data[test_preds == 0, 1],
            color='blue', s=5, alpha=0.5, label='Predicted Class 0 (Blue)')

# Plot predicted class 1 (Red)
plt.scatter(test_data[test_preds == 1, 0], test_data[test_preds == 1, 1],
            color='red', s=5, alpha=0.5, label='Predicted Class 1 (Red)')

# Overlay training points (for reference)
plt.scatter(X_train[train_labels == 0], Y_train[train_labels == 0],
            color='darkblue', edgecolor='black', s=100, label='Train Class 0')
plt.scatter(X_train[train_labels == 1], Y_train[train_labels == 1],
            color='darkred', edgecolor='black', s=100, label='Train Class 1')

plt.title('kNN Classification with k=3 on Test Data (10,000 points)')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
