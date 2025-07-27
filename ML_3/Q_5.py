import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel("SMART_dataset.xlsx")  # Adjust path if needed

# Drop identifier and datetime columns if they exist
drop_cols = ['serial_number']
drop_cols += list(df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns)

# Drop object/string type columns that can't be converted to float
drop_cols += list(df.select_dtypes(include=['object']).columns)

# Drop safely (only those that are present)
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Fill missing values
df = df.fillna(0)

# Separate features and target
X = df.drop(columns=['failure'])
y = df['failure']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train kNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy of kNN (k=3):", accuracy)
