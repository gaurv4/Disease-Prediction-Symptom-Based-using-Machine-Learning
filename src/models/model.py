from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X, y):
    """
    Train a simple Decision Tree classifier.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model
