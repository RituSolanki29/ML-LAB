import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import joblib

# Load cleaned dataset
df = pd.read_excel("DataSet2.xlsx")
if 'date' in df.columns:
    df = df.drop(columns=['date'])

# Prepare the feature matrix (to get feature names for the plot)
target_col = 'failure'
X = df.drop(columns=[target_col])

# Encode categorical columns if any
for col in X.select_dtypes(include=['object','datetime']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Loading the trained model
model = joblib.load('decision_tree_model.joblib')

# Visualizing the tree 
plt.figure(figsize=(12, 8))
plot_tree(model,
          feature_names=X.columns,
          class_names=['0','1'],
          filled=True,
          rounded=True)
plt.title("Decision Tree (loaded model)")
plt.savefig("A6_tree.png")
plt.show()