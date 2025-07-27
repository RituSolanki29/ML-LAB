import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel("SMART_dataset.xlsx")

# Drop identifier column if present
if 'serial_number' in df.columns:
    df = df.drop(columns=['serial_number'])

# Drop datetime columns
datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
df = df.drop(columns=datetime_cols)

# Drop non-numeric columns like model names
non_numeric_cols = df.select_dtypes(include=['object']).columns
df = df.drop(columns=non_numeric_cols)

# Fill missing values
df = df.fillna(0)

# Split into features and target
X = df.drop(columns=['failure'])
y = df['failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Store accuracy results
k_values = range(1, 12)
accuracies = []

# Loop through different k values
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"k={k} → Accuracy: {acc:.4f}")

# Plotting the accuracy
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
plt.title("Accuracy vs k (in kNN)")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()
