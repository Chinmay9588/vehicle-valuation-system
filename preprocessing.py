import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def preprocess_data(df, important_cols=None):
    """Handles missing values and encodes categorical variables."""
    print("Starting preprocessing...")
    
    # 1. Select only important columns if provided
    if important_cols:
        existing_cols = [col for col in important_cols if col in df.columns]
        df = df[existing_cols].copy()
        print(f"Selected {len(existing_cols)} important columns.")
    
    # 2. Handle missing values
    initial_shape = df.shape
    df = df.dropna()
    print(f"Dropped rows with missing values. Shape changed from {initial_shape} to {df.shape}")
    
    if df.empty:
        raise ValueError("Dataset is empty after dropping missing values. Please check your data or column selection.")
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Encoding categorical variables: {categorical_cols}")
    
    # 3. Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    # Save encoders for future use
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    encoders_path = os.path.join(models_dir, 'encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    
    print(f"Categorical variables encoded and encoders saved to {encoders_path}")
    return df