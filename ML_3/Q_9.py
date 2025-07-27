import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_excel("SMART_dataset.xlsx")

# Drop unwanted columns
if 'serial_number' in df.columns:
    df = df.drop(columns=['serial_number'])

# Drop datetime & non-numeric columns
datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
df = df.drop(columns=datetime_cols)
non_numeric_cols = df.select_dtypes(include=['object']).columns
df = df.drop(columns=non_numeric_cols)

# Fill missing values
df = df.fillna(0)

# Split features and target
X = df.drop(columns=['failure'])
y = df['failure']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a kNN model (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on training and test data
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Accuracy
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy :", accuracy_score(y_test, y_test_pred))

# Confusion Matrix
print("\nConfusion Matrix (Train):\n", confusion_matrix(y_train, y_train_pred))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))

# Classification Report (Precision, Recall, F1)
print("\nClassification Report (Train):\n", classification_report(y_train, y_train_pred))
print("Classification Report (Test):\n", classification_report(y_test, y_test_pred))
