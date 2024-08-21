from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter

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