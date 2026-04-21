from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(model, X_test, y_test):
    """Evaluates the model using R2, MAE, and RMSE metrics."""
    print("Evaluating model performance...")
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    metrics = {
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse
    }
    
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    return metrics