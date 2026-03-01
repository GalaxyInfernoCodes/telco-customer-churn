import pandas as pd
import numpy as np

def preprocess_features(df: pd.DataFrame, target_col: str = "Churn", categorical_features: list = None, numerical_features: list = None):
    """
    Preprocesses the input DataFrame for CatBoost training/inference.
    
    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        categorical_features: List of categorical feature names.
        numerical_features: List of numerical feature names.
        
    Returns:
        X: Preprocessed features DataFrame.
        y: Target series (if target_col is in df).
    """
    df = df.copy()
    
    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        
    # Handle TotalCharges: Convert to numeric, coerce errors (empty strings become NaN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0) # Fill NaN with 0 for simplicity
        
    # Prepare X and y
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        # Convert target 'Yes'/'No' to 1/0 if needed, but CatBoost can handle it.
        # Let's verify standard practice. Usually better to map to 0/1 for metrics.
        y = y.map({"Yes": 1, "No": 0})
    else:
        y = None
        X = df
        
    # Ensure categorical features are strings
    # We infer categorical features if not provided, by checking object/category types
    if categorical_features is None:
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
        
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype(str)
            
    return X, y
