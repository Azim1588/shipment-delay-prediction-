"""
Data loading utilities for the Shipment Delay Predictor project.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import DATA_FILE, FEATURES, PROJECT_ROOT

def load_shipment_data(file_path=None):
    """
    Load and preprocess shipment data.
    
    Args:
        file_path (str, optional): Path to the data file. If None, uses default.
    
    Returns:
        pd.DataFrame: Preprocessed data
    """
    if file_path is None:
        file_path = PROJECT_ROOT / DATA_FILE
    
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        print(f"✅ Data loaded successfully: {len(df)} records")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def preprocess_data(df):
    """
    Preprocess the shipment data.
    
    Args:
        df (pd.DataFrame): Raw data
    
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Drop irrelevant columns
    columns_to_drop = [
        'Product Description', 'Customer Email', 'Customer Fname', 'Customer Lname',
        'Customer Password', 'Customer Street', 'Product Image', 'Order Zipcode'
    ]
    
    df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Create target variable
    df_clean['delayed'] = (df_clean['Days for shipping (real)'] > df_clean['Days for shipment (scheduled)']).astype(int)
    
    # Encode categorical columns
    cat_cols = ['Shipping Mode', 'Customer Segment', 'Order Region', 'Order State', 'Market']
    for col in cat_cols:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
    
    # Drop rows with missing data
    df_clean = df_clean.dropna()
    
    # Add date features
    if 'order date (DateOrders)' in df_clean.columns:
        df_clean['order date (DateOrders)'] = pd.to_datetime(df_clean['order date (DateOrders)'])
        df_clean['order_month'] = df_clean['order date (DateOrders)'].dt.to_period('M')
        df_clean['order_week'] = df_clean['order date (DateOrders)'].dt.to_period('W')
    
    print(f"✅ Data preprocessed: {len(df_clean)} records after cleaning")
    return df_clean

def prepare_features(df):
    """
    Prepare features for model training.
    
    Args:
        df (pd.DataFrame): Preprocessed data
    
    Returns:
        tuple: (X, y) features and target
    """
    # Select features
    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features]
    y = df['delayed']
    
    print(f"✅ Features prepared: {X.shape[1]} features, {len(y)} samples")
    return X, y

def get_data_summary(df):
    """
    Get summary statistics for the data.
    
    Args:
        df (pd.DataFrame): Data to summarize
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_records': len(df),
        'delay_rate': df['delayed'].mean() * 100,
        'total_sales': df['Sales'].sum() if 'Sales' in df.columns else 0,
        'avg_sales': df['Sales'].mean() if 'Sales' in df.columns else 0,
        'unique_customers': df['Customer Segment'].nunique() if 'Customer Segment' in df.columns else 0,
        'date_range': None
    }
    
    if 'order date (DateOrders)' in df.columns:
        summary['date_range'] = {
            'start': df['order date (DateOrders)'].min().strftime('%Y-%m-%d'),
            'end': df['order date (DateOrders)'].max().strftime('%Y-%m-%d')
        }
    
    return summary 