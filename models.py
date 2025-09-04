import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    accuracy_score, 
    classification_report,
    mean_absolute_error,
    precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    """
    Preprocess the data for modeling.
    
    Args:
        data (DataFrame): The input data
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, y_type_train, y_type_test
    """
    # Handle missing values for numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    
    # Replace infinite values with NaN and then fill with mean
    data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    
    # Define features and targets
    features = [
        'Carbon Source Quantity (g/L)', 
        'Time (h)', 
        'Temperature (°C)', 
        'pH', 
        'C/N Ratio', 
        'Nitrogen Source Quantity (g/L)'
    ]
    
    # Check if all required columns are present
    missing_cols = [col for col in features if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    X = data[features]
    
    # If PHA Production column exists, use it for regression target
    if 'PHA Production (g/L)' in data.columns:
        y = data['PHA Production (g/L)']
    else:
        # If not available, create a dummy target for demonstration
        y = pd.Series([0] * len(data))
    
    # If PHA Type column exists, use it for classification target
    if 'PHA Type' in data.columns:
        y_type = data['PHA Type']
    else:
        # If not available, create a dummy target for demonstration
        y_type = pd.Series(['Unknown'] * len(data))
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, y_type_train, y_type_test = train_test_split(
        X, y, y_type, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, y_type_train, y_type_test

def train_models(X_train, y_train, y_type_train):
    """
    Train the PHA yield and type prediction models.
    
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Target for yield prediction
        y_type_train (Series): Target for type prediction
        
    Returns:
        tuple: (yield_model, type_model)
    """
    # Train Random Forest Regressor for PHA yield prediction
    yield_model = RandomForestRegressor(n_estimators=200, random_state=42)
    yield_model.fit(X_train, y_train)
    
    # Train Random Forest Classifier for PHA type prediction
    type_model = RandomForestClassifier(n_estimators=200, random_state=42)
    type_model.fit(X_train, y_type_train)
    
    return yield_model, type_model

def predict_pha_yield_and_type(yield_model, type_model, input_data):
    """
    Predict PHA yield and type from input data.
    
    Args:
        yield_model: Trained yield prediction model
        type_model: Trained type prediction model
        input_data (DataFrame): Input data for prediction
        
    Returns:
        tuple: (predicted_yield, predicted_type)
    """
    # Make predictions
    predicted_yield = yield_model.predict(input_data)
    predicted_type = type_model.predict(input_data)
    
    return predicted_yield, predicted_type

def get_model_performance(yield_model, type_model, X_test, y_test, y_type_test):
    """
    Calculate performance metrics for the models.
    
    Args:
        yield_model: Trained yield prediction model
        type_model: Trained type prediction model
        X_test (DataFrame): Test features
        y_test (Series): Test target for yield
        y_type_test (Series): Test target for type
        
    Returns:
        dict: Performance metrics
    """
    # Make predictions
    y_pred = yield_model.predict(X_test)
    y_type_pred = type_model.predict(X_test)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_type_test, y_type_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_type_test, y_type_pred, average='macro'
    )
    class_report = classification_report(y_type_test, y_type_pred)
    
    # Return all metrics
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': class_report,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_type_test': y_type_test,
        'y_type_pred': y_type_pred
    }

def get_feature_importance(model, feature_names):
    """
    Get feature importance from the trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        
    Returns:
        DataFrame: Feature importance data
    """
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df
