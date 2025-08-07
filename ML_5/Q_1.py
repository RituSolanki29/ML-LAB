import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel('SMART_dataset.xlsx')

# Select two numerical SMART features (use any pair you want)
X = df[['smart_5_raw']]         
y = df['smart_9_raw']            

# Drop missing values
data = pd.concat([X, y], axis=1).dropna()
X = data[['smart_5_raw']]
y = data['smart_9_raw']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
reg = LinearRegression().fit(X_train, y_train)

# Predict
y_train_pred = reg.predict(X_train)

# Plot
plt.scatter(X_train, y_train, color='blue', label='Actual')
plt.plot(X_train, y_train_pred, color='red', label='Predicted')
plt.xlabel("smart_5_raw")
plt.ylabel("smart_9_raw")
plt.title("Linear Regression: smart_5_raw vs smart_9_raw")
plt.legend()
plt.show()

# Print coefficients
print("Coefficient:", reg.coef_)
print("Intercept:", reg.intercept_)
