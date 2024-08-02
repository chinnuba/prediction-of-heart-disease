import matplotlib.pyplot as plt
import seaborn as sns

def summarize_dataset(df):
    # Summarize the dataset
    print("\nDataset Summary:")
    print(df.info())
    print("\nFirst 5 Rows of the Dataset:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

def visualize_target_distribution(df, target_column):
    # Visualize the distribution of the target variable
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_column, data=df)
    plt.title(f'Distribution of {target_column}')
    plt.show()

def visualize_feature_distribution(df):
    # Visualize the distribution of each feature in the dataset
    df.hist(figsize=(20, 15))
    plt.suptitle('Feature Distributions', fontsize=20)
    plt.show()

def correlation_analysis(df):
    # Perform correlation analysis between numerical features and the target variable
    correlation_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

def feature_interaction_analysis(df, categorical_columns, target_column):
    # Analyze interactions between categorical features and the target variable
    for column in categorical_columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=column, hue=target_column, data=df)
        plt.title(f'{column} vs {target_column}')
        plt.show()

def outlier_detection(df, numerical_columns):
    # Detect and visualize outliers in the dataset
    for column in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=column, data=df)
        plt.title(f'Boxplot of {column}')
        plt.show()

def distribution_analysis_by_target(df, numerical_columns, target_column):
    # Compare the distribution of numerical features across different target classes
    for column in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=column, hue=target_column, kde=True, element='step')
        plt.title(f'Distribution of {column} by {target_column}')
        plt.show()
