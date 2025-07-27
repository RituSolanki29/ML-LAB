import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel("SMART_dataset.xlsx")  

# Drop identifier column if it exists
if 'serial_number' in df.columns:
    df = df.drop(columns=['serial_number'])

# Drop datetime columns if any
datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
df = df.drop(columns=datetime_cols)

# Drop non-numeric columns (like model names)
non_numeric_cols = df.select_dtypes(include=['object']).columns
df = df.drop(columns=non_numeric_cols)

# Fill missing values with 0
df = df.fillna(0)

# Separate features and target
X = df.drop(columns=['failure'])
y = df['failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train kNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy of kNN (k=3):", accuracy)

# Score using .score()
score = knn.score(X_test, y_test)
print("Score using knn.score():", score)

# Predictions for full test set
all_predictions = knn.predict(X_test)
print("\nPredictions for entire test set:")
print(all_predictions)

# Predict a single test vector
sample_index = 0  
test_vect = X_test.iloc[sample_index].values.reshape(1, -1)
predicted_class = knn.predict(test_vect)

print(f"\nPrediction for test vector at index {sample_index}: {predicted_class[0]}")
print(f"Actual class: {y_test.iloc[sample_index]}")
