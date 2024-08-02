import pandas as pd
from src.model_evaluation import evaluate_model

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test, data_label="Original Data"):
    """
    Train and evaluate models, and print the results.

    Parameters:
    - models: Dictionary of model names and model instances.
    - X_train: Training features.
    - X_test: Testing features.
    - y_train: Training target.
    - y_test: Testing target.
    - data_label: Label to describe the dataset (e.g., "Original Data" or "After SMOTE").

    Returns:
    - pd.DataFrame: DataFrame containing evaluation metrics for each model.
    """
    results = {}
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name} on {data_label}...")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[model_name] = metrics

    results_df = pd.DataFrame(results).T
    print(f"\nModel Performance Comparison on {data_label}:")
    print(results_df)

    return results_df

def compare_results(original_results_df, smote_results_df):
    """
    Compare the results before and after applying SMOTE.

    Parameters:
    - original_results_df: DataFrame containing results from original data.
    - smote_results_df: DataFrame containing results from SMOTE data.

    Returns:
    - pd.DataFrame: DataFrame showing the differences in metrics before and after SMOTE.
    """
    comparison_df = smote_results_df.copy()
    comparison_df['Accuracy Change'] = smote_results_df['Accuracy'] - original_results_df['Accuracy']
    comparison_df['Precision Change'] = smote_results_df['Precision'] - original_results_df['Precision']
    comparison_df['Recall Change'] = smote_results_df['Recall'] - original_results_df['Recall']
    comparison_df['F1 Score Change'] = smote_results_df['F1 Score'] - original_results_df['F1 Score']
    comparison_df['ROC AUC Change'] = smote_results_df['ROC AUC'] - original_results_df['ROC AUC']

    print("\nComparison of Model Performance")
    print(comparison_df)

    return comparison_df
