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

# Drop string/object columns (non-numeric)
string_cols = df.select_dtypes(include=['object']).columns
df = df.drop(columns=string_cols)

# Fill missing values with 0
df = df.fillna(0)

# Separate features and target
X = df.drop(columns=['failure'])
y = df['failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train kNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy of kNN (k=3) using accuracy_score():", accuracy)

# Also using built-in score method
print("Test Accuracy using knn.score():", knn.score(X_test, y_test))
