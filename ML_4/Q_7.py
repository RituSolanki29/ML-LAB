import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load your project data
df = pd.read_excel('DataSet2.xlsx', engine='openpyxl')  

# Select two relevant features and the target column
X = df[['smart_5_raw', 'smart_187_raw']]  
y = df['failure']  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
knn = KNeighborsClassifier()

# Set up the parameter grid (range of k values to test)
param_grid = {'n_neighbors': list(range(1, 21))}

# Use GridSearchCV to find the best k
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Output the best parameters and best score
print(" Best k value found:", grid_search.best_params_['n_neighbors'])
print(" Best cross-validation accuracy:", round(grid_search.best_score_, 4))
