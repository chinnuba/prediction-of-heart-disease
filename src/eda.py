from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def feature_importance_analysis(X_train, y_train, feature_names):
    # Perform feature importance analysis using a Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 8))
    plt.title("Feature Importance Analysis")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()

def perform_pca(X_train, y_train, n_components=2):
    # Perform Principal Component Analysis (PCA) and plot the results
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_train)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y_train, cmap='viridis', edgecolor='k')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA: First Two Principal Components')
    plt.colorbar(scatter, label='Target')
    plt.grid()
    plt.show()
