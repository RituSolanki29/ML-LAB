import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel('SMART_dataset.xlsx')

# Select multiple numerical features and one numerical target
feature_cols = ['smart_5_raw', 'smart_187_raw', 'smart_197_raw', 'smart_198_raw']  # choose any available numeric features
target_col = 'smart_9_raw'

# Filter and drop missing values
data = df[feature_cols + [target_col]].dropna()
X = data[feature_cols]
y = data[target_col]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
reg = LinearRegression().fit(X_train, y_train)

# Predict
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

def print_metrics(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nMetrics for {label}:")
    print(f"  MSE  = {mse:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAPE = {mape:.4f}")
    print(f"  R2   = {r2:.4f}")

print_metrics(y_train, y_train_pred, "Training Set (Multiple Features)")
print_metrics(y_test, y_test_pred, "Test Set (Multiple Features)")

plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.xlabel("Actual smart_9_raw")
plt.ylabel("Predicted smart_9_raw")
plt.title("Predicted vs Actual (Test Set)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.show()
