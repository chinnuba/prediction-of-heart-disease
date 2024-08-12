from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_models():
    """
    Define and return a dictionary of models to be trained and evaluated.

    Returns:
    - dict: A dictionary containing model names and their corresponding instances.
    """
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'Custom Neural Network': MLPClassifier(
            hidden_layer_sizes=(64, 128, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=64,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
    }
    return models
