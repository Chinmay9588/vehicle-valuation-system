import pandas as pd
import os

def load_data(file_path):
    """Loads dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df