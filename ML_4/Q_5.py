import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate training data (same as before)
np.random.seed(0)
X_train = np.random.uniform(1, 10, 20)
Y_train = np.random.uniform(1, 10, 20)
train_data = np.column_stack((X_train, Y_train))
train_labels = np.where(X_train + Y_train > 10, 1, 0)

# Generate test grid (100 x 100 = 10,000 points)
x_test_vals = np.arange(0, 10.1, 0.1)
y_test_vals = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_test_vals, y_test_vals)
test_data = np.c_[xx.ravel(), yy.ravel()]

# List of k values to observe the effect
k_values = [1, 3, 5, 7, 9]

plt.figure(figsize=(20, 12))

# Iterate over k values
for idx, k in enumerate(k_values, 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    test_preds = knn.predict(test_data)

    # Plotting
    plt.subplot(2, 3, idx)
    plt.scatter(test_data[test_preds == 0, 0], test_data[test_preds == 0, 1],
                color='blue', s=5, alpha=0.5, label='Class 0')
    plt.scatter(test_data[test_preds == 1, 0], test_data[test_preds == 1, 1],
                color='red', s=5, alpha=0.5, label='Class 1')

    # Overlay training points
    plt.scatter(X_train[train_labels == 0], Y_train[train_labels == 0],
                color='darkblue', edgecolor='black', s=100, label='Train Class 0')
    plt.scatter(X_train[train_labels == 1], Y_train[train_labels == 1],
                color='darkred', edgecolor='black', s=100, label='Train Class 1')

    plt.title(f'kNN Decision Boundary (k={k})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='upper right')
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Effect of Varying k in kNN on Class Boundaries", fontsize=16, y=1.02)
plt.show()
