import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_excel("SMART_dataset.xlsx")  

# Drop non-numeric and identifier columns if needed
if 'serial_number' in df.columns:
    df = df.drop(columns=['serial_number'])

# Drop rows with missing values (or fill them if preferred)
df = df.fillna(0)


# Separate features and labels
X = df.drop(columns=['failure'])  
y = df['failure']                 

# Split into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print shape for confirmation
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
