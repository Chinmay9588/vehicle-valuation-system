from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def train_model(X_train, y_train):
    """Trains a RandomForestRegressor model with fewer estimators to save memory."""
    print("Training RandomForestRegressor model (Optimized for memory)...")
    # Using 20 estimators instead of 100 to reduce model size / memory usage
    model = RandomForestRegressor(n_estimators=20, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training successful.")
    return model

def save_model(model, file_path):
    """Saves the trained model to a pkl file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully to {file_path}")