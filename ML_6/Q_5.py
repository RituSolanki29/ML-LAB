import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_excel("DataSet2.xlsx")

# Drop date column if present
if 'date' in df.columns:
    df = df.drop(columns=['date'])

target_col = 'failure'
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode any remaining categorical columns
for col in X.select_dtypes(include=['object','datetime']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dt_model.fit(X, y)

# Saving the model for later use
model_path = "decision_tree_model.joblib"
joblib.dump(dt_model, model_path)

print("Decision Tree model created.")