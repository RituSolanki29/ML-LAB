import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Load dataset
df = pd.read_excel('DataSet2.xlsx')  

X = df[['smart_5_normalized', 'smart_9_normalized', 'smart_194_normalized']].values  
y = df['failure'].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# 1. Predict labels for the entire test set
y_pred = neigh.predict(X_test)
print("Predictions for all test vectors:")
print(y_pred)

# 2. Predict label for a specific test vector
test_vect = X_test[0]  # You can choose any test vector
predicted_class = neigh.predict([test_vect])[0]
print(f"\nPrediction for test vector {test_vect}: Class {predicted_class}")