import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the Excel sheet
df = pd.read_excel('Lab Session Data.xlsx', sheet_name="Purchase data", engine='openpyxl')

print("Data Preview:")
print(df)


# Step 1: Label as RICH (1) or POOR (0)
df['Label'] = df['Payment (Rs)'].apply(lambda x: 1 if x > 200 else 0)

# Step 2: Define features (X) and target (y)
X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
y = df['Label']

# Step 3: Split data into train and test sets (e.g., 70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train Logistic Regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["POOR", "RICH"]))

# Optional: Predict new customer behavior
new_data = pd.DataFrame({
    'Candies (#)': [18],
    'Mangoes (Kg)': [4],
    'Milk Packets (#)': [2]
})
pred = model.predict(new_data)
print("\nPrediction for new customer:", "RICH" if pred[0] == 1 else "POOR")
