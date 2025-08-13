import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


# Minimal DecisionTree for demonstration
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):
        # This is a dummy tree for demonstration purposes
        # Replace with your actual tree-building logic
        return {
            "feature": X.columns[0],
            "info_gain": 0.5,
            "branches": {
                0: {"is_leaf": True, "prediction": 0},
                1: {
                    "feature": X.columns[1],
                    "info_gain": 0.3,
                    "branches": {
                        0: {"is_leaf": True, "prediction": 1},
                        1: {"is_leaf": True, "prediction": 0}
                    }
                }
            }
        }


def visualize_tree(tree, parent_name, graph, node_id=0):
    """
    Recursive function to visualize the decision tree.
    """
    # If leaf node
    if tree.get("is_leaf", False):
        node_label = f"Leaf: {tree['prediction']}"
        graph.add_node(node_id, label=node_label, color="lightgreen")
        return node_id

    # If internal node
    feature = tree.get("feature", "Unknown")
    info_gain = tree.get("info_gain", 0)
    node_label = f"Feature: {feature}\nIG: {info_gain:.2f}"
    graph.add_node(node_id, label=node_label, color="lightblue")

    # Add edges for each value of the feature
    for value, subtree in tree.get("branches", {}).items():
        child_id = len(graph.nodes)
        child_id = visualize_tree(subtree, feature, graph, child_id)
        graph.add_edge(node_id, child_id, label=str(value))

    return node_id


def plot_tree(tree):
    """
    Build and plot the decision tree using NetworkX.
    """
    G = nx.DiGraph()
    visualize_tree(tree, None, G)

    pos = nx.spring_layout(G, seed=42)
    labels = nx.get_node_attributes(G, "label")
    edge_labels = nx.get_edge_attributes(G, "label")

    nx.draw(G, pos, with_labels=False, node_size=3000, node_color="lightgray", font_size=8)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title("Decision Tree Visualization")
    plt.show()


# ---------------- Example usage ----------------
if __name__ == "__main__":
    df = pd.read_excel("DataSet2.xlsx")
    X = df.drop(columns=["failure"])
    y = df["failure"]

    tree = DecisionTree(max_depth=3)
    trained_tree = tree.fit(X, y)

    plot_tree(trained_tree)
