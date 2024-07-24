import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report

def _initialize_models():
    """
    Initialize the machine learning models to be compared.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC()
    }
    return models

def _cross_validate_models(models, X_train, y_train):
    """
    Perform cross-validation on the provided models and return the results.
    """
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = scores.mean()
        print(f'{name}: Mean Accuracy = {scores.mean():.4f}')
    return results

def _hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning for the Random Forest model and return the best model.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f'Best parameters for Random Forest: {grid_search.best_params_}')
    return grid_search.best_estimator_

def _evaluate_model(model, X_test, y_test):
    """
    Evaluate the provided model on the test data and print the classification report.
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Initialize models
    models = _initialize_models()

    # Cross-validate models
    _cross_validate_models(models, X_train, y_train)

    # Hyperparameter tuning for Random Forest
    best_rf_model = _hyperparameter_tuning(X_train, y_train)

    # Evaluate the best Random Forest model
    print("\nEvaluation of the best Random Forest model on the test set:")
    _evaluate_model(best_rf_model, X_test, y_test)
