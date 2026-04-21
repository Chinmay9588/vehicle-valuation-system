import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import required modules from src
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import perform_feature_engineering
from src.train_model import train_model, save_model
from src.evaluate import evaluate_model

def estimate_present_price(df):
    """
    Estimates the ex-showroom price (present_price) for each car.
    
    Since the dataset doesn't have a direct 'present_price' column, we estimate it
    using a reverse-depreciation approach:
        present_price = selling_price / depreciation_factor(car_age)
    
    This gives the model a realistic present_price feature to learn from.
    """
    import datetime
    current_year = datetime.datetime.now().year
    
    # Calculate car age
    year_col = None
    for col in ['year', 'myear', 'Year']:
        if col in df.columns:
            year_col = col
            break
    
    if year_col is None:
        raise ValueError("Year column not found in dataset.")
    
    car_age = current_year - df[year_col]
    
    # Standard depreciation curve for Indian cars (approximate)
    # Year 0: 100%, Year 1: 80%, Year 2: 70%, Year 3: 60%, ...
    # Formula: retained_value = 0.85 * (0.90 ^ car_age), capped at a minimum of 10%
    retained_fraction = 0.85 * (0.90 ** car_age)
    retained_fraction = retained_fraction.clip(lower=0.10)  # At least 10% retained
    
    # Estimate present_price: what the car would cost new
    # selling_price = present_price * retained_fraction
    # => present_price = selling_price / retained_fraction
    target_col = None
    for col in ['selling_price', 'listed_price']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError("Selling price column not found in dataset.")
    
    df['present_price'] = (df[target_col] / retained_fraction).round(0)
    
    # Convert to lakhs for consistency (model internally uses raw values)
    # Keep as raw rupees since the UI will convert lakhs → rupees
    
    print(f"Estimated 'present_price' for {len(df)} records using depreciation model.")
    print(f"  Present price range: Rs.{df['present_price'].min():,.0f} - Rs.{df['present_price'].max():,.0f}")
    print(f"  Mean present price:  Rs.{df['present_price'].mean():,.0f}")
    
    return df

def main():
    try:
        # Configuration
        DATA_PATH = os.path.join('data', 'processed', 'cars_data_clean.csv')
        MODEL_SAVE_PATH = os.path.join('models', 'vehicle_price_model.pkl')
        TARGET_COLUMN = 'selling_price'
        
        # Define columns we actually need to load to save memory
        COLS_TO_LOAD = ['model', 'myear', 'listed_price', 'km', 'fuel', 'transmission', 'owner_type']

        print("--- Vehicle Price Prediction Pipeline Started ---")

        # 2. Load dataset (optimized with usecols)
        print(f"Loading only relevant columns: {COLS_TO_LOAD}")
        raw_df = pd.read_csv(DATA_PATH, usecols=COLS_TO_LOAD, low_memory=False)
        print(f"Dataset loaded. Initial shape: {raw_df.shape}")

        # Create 'name' column and extract Brand
        raw_df['name'] = raw_df['model'].fillna('unknown unknown')
        raw_df['Brand'] = raw_df['name'].apply(lambda x: str(x).split()[0].lower())
        print("Brand extracted from car name.")

        column_mapping = {
            'myear': 'year',
            'listed_price': 'selling_price',
            'km': 'km_driven',
            'fuel': 'fuel',
            'transmission': 'transmission',
            'owner_type': 'owner',
            'Brand': 'Brand'
        }
        
        # Select and rename columns
        df = raw_df[list(column_mapping.keys())].rename(columns=column_mapping)
        print(f"Features selected: {df.columns.tolist()}")

        # 3. Estimate present_price (ex-showroom price) before preprocessing
        df = estimate_present_price(df)

        # 4. Perform preprocessing (handles missing values + encodes categoricals)
        df = preprocess_data(df)

        # 5. Perform feature engineering (creates Car_Age, drops year column)
        df = perform_feature_engineering(df)

        # Ensure consistent feature order:
        # present_price, Brand, km_driven, fuel, transmission, owner, Car_Age
        final_feature_order = ['present_price', 'Brand', 'km_driven', 'fuel', 'transmission', 'owner', 'Car_Age', TARGET_COLUMN]
        df = df[final_feature_order]

        # 6. Split dataset into train/test sets
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        print(f"\nTraining on features: {X.columns.tolist()}")
        print(f"Target: {TARGET_COLUMN}")
        print(f"Dataset size: {len(X)} samples")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 7. Train model using RandomForestRegressor
        model = train_model(X_train, y_train)

        # 8. Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # 9. Save trained model
        save_model(model, MODEL_SAVE_PATH)

        print("\n--- Pipeline Completed Successfully ---")
        print(f"\nFeature order for prediction: {X.columns.tolist()}")

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        print("Tip: If out of memory, try reducing columns or using chunks.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()