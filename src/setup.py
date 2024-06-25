import pandas as pd

def load_dataset(filepath):
    """
    Load the dataset using pandas.
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded from {filepath}.")
    return df