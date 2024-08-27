from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def _check_class_imbalance(y_train):
    """
    Check and visualize class distribution before applying SMOTE.

    Parameters:
    - y_train: Training target.

    Returns:
    - None
    """
    counter = Counter(y_train)
    print(f"Class distribution before SMOTE: {counter}")
    plt.bar(counter.keys(), counter.values())
    plt.title('Class Distribution Before SMOTE')
    plt.show()

def apply_smote(X_train, y_train):
    """
    Apply SMOTE to handle class imbalance.

    Parameters:
    - X_train: Training features.
    - y_train: Training target.

    Returns:
    - X_resampled: Resampled training features.
    - y_resampled: Resampled training target.
    """
    _check_class_imbalance(y_train)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    counter = Counter(y_resampled)
    print(f"Class distribution after SMOTE: {counter}")
    plt.bar(counter.keys(), counter.values())
    plt.title('Class Distribution After SMOTE')
    plt.show()
    return X_resampled, y_resampled

def hyperparameter_tuning_neural_network(X_train, y_train):
    param_grid = {
        'hidden_layer_sizes': [(64, 128, 64), (128, 128, 128), (64, 128, 128, 64)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'batch_size': [32, 64, 128]
    }

    mlp = MLPClassifier(max_iter=500, random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params

def hyperparameter_tuning_random_forest(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    return rf_random.best_estimator_, rf_random.best_params_

def hyperparameter_tuning_neural_network(X_train, y_train):
    param_dist = {
        'hidden_layer_sizes': [(64, 64), (128, 128), (128, 256, 128)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [300, 500, 700]
    }

    mlp = MLPClassifier(random_state=42)
    mlp_random = RandomizedSearchCV(estimator=mlp, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, n_jobs=-1)
    mlp_random.fit(X_train, y_train)
    return mlp_random.best_estimator_, mlp_random.best_params_

def hyperparameter_tuning_decision_tree(X_train, y_train):
    param_dist = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    dt = DecisionTreeClassifier(random_state=42)
    dt_random = RandomizedSearchCV(estimator=dt, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    dt_random.fit(X_train, y_train)
    return dt_random.best_estimator_, dt_random.best_params_

def hyperparameter_tuning_svm(X_train, y_train):
    param_dist = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    svm = SVC(random_state=42, probability=True)
    svm_random = RandomizedSearchCV(estimator=svm, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    svm_random.fit(X_train, y_train)
    return svm_random.best_estimator_, svm_random.best_params_

def hyperparameter_tuning_logistic_regression(X_train, y_train):
    param_dist = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2']
    }

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_random = RandomizedSearchCV(estimator=lr, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    lr_random.fit(X_train, y_train)
    return lr_random.best_estimator_, lr_random.best_params_

def hyperparameter_tuning_knn(X_train, y_train):
    param_dist = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    knn = KNeighborsClassifier()
    knn_random = RandomizedSearchCV(estimator=knn, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    knn_random.fit(X_train, y_train)
    return knn_random.best_estimator_, knn_random.best_params_