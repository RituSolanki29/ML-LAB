import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix, mean_squared_error, r2_score
)

# Load dataset
df = pd.read_excel('DataSet2.xlsx')  

target_column = 'failure'
feature_columns = ['smart_5_normalized', 'smart_9_normalized', 'smart_194_normalized']  

# Drop rows with NaN in features or target
df = df[feature_columns + [target_column]].dropna()

# Filter to only two classes (e.g., 0 and 1)
allowed_classes = [0, 1]  
df = df[df[target_column].isin(allowed_classes)]

# Separate features and target
X = df[feature_columns].values
y = df[target_column].values

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train kNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, y_pred, zero_division=0), 4))

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
epsilon = 1e-10
mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))
print("Mean Absolute Percentage Error (MAPE):", round(mape, 2), "%")
print("R2 Score:", round(r2, 4))
