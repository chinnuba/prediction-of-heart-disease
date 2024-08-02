import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def encode_categorical_features(df, categorical_columns):
    # Encode categorical features using One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    print("Categorical features encoded successfully.")
    return df_encoded

def scale_numerical_features(df, numerical_columns):
    # Normalize or standardize numerical features
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    print("Numerical features scaled successfully.")
    return df

def handle_outliers(df, numerical_columns):
    # Handle outliers by capping them to a certain threshold
    for column in numerical_columns:
        upper_limit = df[column].mean() + 3 * df[column].std()
        lower_limit = df[column].mean() - 3 * df[column].std()
        df[column] = np.clip(df[column], lower_limit, upper_limit)
    print("Outliers handled successfully.")
    return df

def split_dataset(df, target_column, test_size=0.2, random_state=42):
    # Split the dataset into training and testing sets
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("Dataset split into training and testing sets.")
    return X_train, X_test, y_train, y_test

def preprocess_data(df, categorical_columns, numerical_columns, target_column):
    # Full preprocessing pipeline for the dataset
    df = encode_categorical_features(df, categorical_columns)
    df = scale_numerical_features(df, numerical_columns)
    df = handle_outliers(df, numerical_columns)
    X_train, X_test, y_train, y_test = split_dataset(df, target_column)
    return X_train, X_test, y_train, y_test
