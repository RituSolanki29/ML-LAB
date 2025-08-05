import pandas as pd
from sklearn.model_selection import train_test_split

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

# Output shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
