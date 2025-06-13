"""
Model utilities for the Shipment Delay Predictor project.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import MODELS_DIR, FEATURES

def load_model(model_path):
    """
    Load a trained model from file.
    
    Args:
        model_path (str): Path to the model file
    
    Returns:
        object: Loaded model
    """
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def save_model(model, model_path):
    """
    Save a trained model to file.
    
    Args:
        model: Trained model to save
        model_path (str): Path where to save the model
    """
    try:
        # Ensure directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"✅ Model saved successfully: {model_path}")
    except Exception as e:
        raise Exception(f"Error saving model: {str(e)}")

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name (str): Name of the model for reporting
    
    Returns:
        dict: Performance metrics
    """
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"📊 {model_name} Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        
        return metrics
    except Exception as e:
        raise Exception(f"Error evaluating model: {str(e)}")

def predict_with_proper_features(model, features_data, feature_names=None):
    """
    Make predictions with proper feature handling.
    
    Args:
        model: Trained model
        features_data: Input features (list or array)
        feature_names: Feature names (optional)
    
    Returns:
        tuple: (predictions, probabilities)
    """
    try:
        if feature_names is None:
            feature_names = FEATURES
        
        # Create DataFrame with proper feature names
        if isinstance(features_data, list):
            features_data = [features_data]
        
        features_df = pd.DataFrame(features_data, columns=feature_names)
        
        # Make predictions
        predictions = model.predict(features_df)
        probabilities = model.predict_proba(features_df)
        
        return predictions, probabilities
    except Exception as e:
        raise Exception(f"Error making predictions: {str(e)}")

def calculate_risk_score(delay_probability, confidence=0.95):
    """
    Calculate risk score based on delay probability.
    
    Args:
        delay_probability (float): Probability of delay
        confidence (float): Confidence level
    
    Returns:
        tuple: (risk_score, risk_level, color)
    """
    # Base risk score (0-100)
    base_score = delay_probability * 100
    
    # Confidence adjustment
    if delay_probability > 0.8:
        confidence_multiplier = 1.2
    elif delay_probability > 0.6:
        confidence_multiplier = 1.1
    elif delay_probability > 0.4:
        confidence_multiplier = 1.0
    else:
        confidence_multiplier = 0.9
    
    risk_score = min(100, base_score * confidence_multiplier)
    
    # Risk level classification
    if risk_score >= 70:
        risk_level = "HIGH"
        color = "red"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
        color = "orange"
    else:
        risk_level = "LOW"
        color = "green"
    
    return risk_score, risk_level, color

def calculate_cost_impact(delay_probability, sales_amount, shipping_cost_ratio=0.1):
    """
    Calculate potential cost impact of delays.
    
    Args:
        delay_probability (float): Probability of delay
        sales_amount (float): Sales amount
        shipping_cost_ratio (float): Shipping cost as ratio of sales
    
    Returns:
        dict: Cost breakdown
    """
    # Estimated costs
    shipping_cost = sales_amount * shipping_cost_ratio
    delay_penalty = sales_amount * 0.05  # 5% penalty for delays
    customer_satisfaction_cost = sales_amount * 0.02  # 2% for customer service
    
    # Expected cost impact
    expected_delay_cost = delay_probability * (delay_penalty + customer_satisfaction_cost)
    total_cost = shipping_cost + expected_delay_cost
    
    return {
        'shipping_cost': shipping_cost,
        'expected_delay_cost': expected_delay_cost,
        'total_cost': total_cost,
        'cost_savings_potential': expected_delay_cost * 0.3  # 30% potential savings
    }

def get_feature_importance(model, feature_names=None):
    """
    Get feature importance from model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
    
    Returns:
        dict: Feature importance scores
    """
    if feature_names is None:
        feature_names = FEATURES
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            return feature_importance
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not extract feature importance: {str(e)}")
        return None 