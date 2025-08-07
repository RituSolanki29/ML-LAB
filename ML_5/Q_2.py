from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel('SMART_dataset.xlsx')

# Select two numerical SMART attributes
X = df[['smart_5_raw']]         # Feature
y = df['smart_9_raw']           # Target

# Drop missing values
data = pd.concat([X, y], axis=1).dropna()
X = data[['smart_5_raw']]
y = data['smart_9_raw']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
reg = LinearRegression().fit(X_train, y_train)

# Predict
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# --- Metrics ---
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

print_metrics(y_train, y_train_pred, "Training Set")
print_metrics(y_test, y_test_pred, "Test Set")

# Optional: Plot
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_test_pred, color='red', label='Predicted')
plt.xlabel("smart_5_raw")
plt.ylabel("smart_9_raw")
plt.title("Linear Regression on Test Set")
plt.legend()
plt.show()
