import datetime
import pandas as pd

def perform_feature_engineering(df):
    """Creates new features and selects important ones."""
    print("Performing feature engineering...")
    
    # 1. Create Car_Age feature
    current_year = datetime.datetime.now().year
    
    # Check for possible year column names
    year_col = None
    for col in ['myear', 'year', 'Year']:
        if col in df.columns:
            year_col = col
            break
            
    if year_col:
        df['Car_Age'] = current_year - df[year_col]
        print(f"Created 'Car_Age' feature using '{year_col}' column.")
        # Optionally drop the original year column
        df = df.drop(columns=[year_col])
    else:
        print("Warning: Year column not found. Skipping Car_Age creation.")

    # 2. Select important features
    # Based on general used car datasets, these are often key:
    # km, fuel, transmission, owner_type (encoded), Car_Age
    # We will keep all columns for now as "important features" or filter if we know the target
    
    # In a real scenario, this would involve correlation analysis or domain knowledge.
    # For this project, we'll assume the input df already has relevant columns.
    
    return df