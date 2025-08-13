import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# A7. Decision Boundary Visualization
def visualize_decision_boundary(df, feature1, feature2, target="failure", max_depth=3):
    """
    Visualize decision boundary using 2 features of dataset with Decision Tree
    """
    # Extract 2 features
    X = df[[feature1, feature2]].values
    y = df[target].values

    # Train Decision Tree
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)

    # Create meshgrid for plotting decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict for each point in meshgrid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f"Decision Boundary using {feature1} & {feature2}")
    plt.show()


if __name__ == "__main__":
    # Load dataset
    df = pd.read_excel("DataSet2.xlsx")

    # Automatically select first two numeric features
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    feature1, feature2 = numeric_cols[0], numeric_cols[1]

    # Visualize decision boundary
    visualize_decision_boundary(df, feature1=feature1, feature2=feature2, target="failure", max_depth=3)
