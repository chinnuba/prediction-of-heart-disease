import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

def plot_training_history(model):
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 4))
        plt.plot(model.loss_curve_)
        plt.title(f'Training Loss Curve for {model.__class__.__name__}')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()

def plot_confusion_matrix_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model.__class__.__name__}")
    plt.show()

def plot_roc_curve_model(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model.__class__.__name__}')
    plt.legend(loc='lower right')
    plt.show()

def plot_feature_importance_model(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 6))
        plt.title(f'Feature Importances for {model.__class__.__name__}')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

def bias_variance_analysis(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Model: {model.__class__.__name__}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Assess bias-variance trade-off
    if abs(train_accuracy - test_accuracy) > 0.05:
        print("There may be a bias-variance trade-off issue (overfitting or underfitting).\n")
    else:
        print("The model seems to have a good balance between bias and variance.\n")

def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train and evaluate a machine learning model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred)
    }

    print(f"Model: {model.__class__.__name__}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

    # Plot the training history
    plot_training_history(model)

    # Plot confusion matrix
    plot_confusion_matrix_model(model, X_test, y_test)

    # Plot ROC curve
    plot_roc_curve_model(model, X_test, y_test)

    # Plot feature importance if applicable
    plot_feature_importance_model(model, X_train.columns)

    # Perform bias-variance analysis
    bias_variance_analysis(model, X_train, X_test, y_train, y_test)

    return metrics
