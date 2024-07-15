from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def _clean_data(df):
    # Correct typos in column names
    df.rename(columns={'cholestoral': 'cholesterol'}, inplace=True)

    # Convert categorical columns to lowercase for consistency
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].str.lower()

    # Convert binary categorical columns to 0 and 1
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['fasting_blood_sugar'] = df['fasting_blood_sugar'].map({'greater than 120 mg/ml': 1, 'lower than 120 mg/ml': 0})
    df['exercise_induced_angina'] = df['exercise_induced_angina'].map({'yes': 1, 'no': 0})

    return df

def _preprocess_data(X, y):
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Define preprocessing for numerical data (impute and scale)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for categorical data (impute and one-hot encode)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Apply the preprocessing to the data
    X = preprocessor.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def _analyze_outliers(df):
    """Analyze each column for potential outliers."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        print(f'\nAnalysis of {col}:')
        print(f'Lower Bound: {lower_bound}, Upper Bound: {upper_bound}')
        print(f'Number of Potential Outliers: {len(outliers)}')
        print(outliers[col].sort_values().to_string(index=False))

def clean_and_preprocess(df):

    # Step 1: Data Cleaning
    df = _clean_data(df)

    # Analyze Outliers
    _analyze_outliers(df)

    # Separate features and target
    X = df.drop(columns='target')
    y = df['target']

    # Step 2: Preprocessing
    X_train, X_test, y_train, y_test = _preprocess_data(X, y)

    return X_train, X_test, y_train, y_test
