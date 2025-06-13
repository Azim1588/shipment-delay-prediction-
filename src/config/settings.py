"""
Configuration settings for the Shipment Delay Predictor project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Data settings
DATA_FILE = "DataCoSupplyChainDataset.csv"
MODEL_FILE = "random_forest_delay_model.pkl"
ENSEMBLE_MODEL_FILE = "ensemble_model.pkl"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature settings
FEATURES = [
    'Shipping Mode', 'Customer Segment', 'Order Region', 'Order State',
    'Days for shipment (scheduled)', 'Order Item Quantity',
    'Order Item Discount Rate', 'Order Item Profit Ratio', 'Sales', 'Order Item Total'
]

# Email settings
EMAIL_CONFIG = {
    'sender': os.getenv('EMAIL_SENDER', 'your-email@gmail.com'),
    'password': os.getenv('EMAIL_PASSWORD', 'your-app-password'),
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', 587))
}

# Streamlit settings
STREAMLIT_CONFIG = {
    'page_title': "🚚 Shipment Delay Predictor",
    'page_icon': "🚚",
    'layout': "wide"
}

# Model performance thresholds
PERFORMANCE_THRESHOLDS = {
    'accuracy_min': 0.65,
    'precision_min': 0.80,
    'recall_min': 0.50,
    'f1_min': 0.60
}

# Risk scoring
RISK_THRESHOLDS = {
    'low': 40,
    'medium': 70,
    'high': 100
} 