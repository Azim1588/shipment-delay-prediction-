"""
Model training module for the Shipment Delay Predictor project.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import RANDOM_STATE, TEST_SIZE, CV_FOLDS, FEATURES
from ..utils.data_loader import load_shipment_data, preprocess_data, prepare_features
from ..utils.model_utils import save_model, evaluate_model

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class ModelTrainer:
    """Main model trainer class."""
    
    def __init__(self, data_path=None):
        """
        Initialize the model trainer.
        
        Args:
            data_path (str, optional): Path to the data file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for training."""
        print("🚚 Loading and preparing data...")
        
        # Load data
        self.df = load_shipment_data(self.data_path)
        
        # Preprocess data
        self.df = preprocess_data(self.df)
        
        # Prepare features
        X, y = prepare_features(self.df)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"✅ Data prepared: {self.X_train.shape[0]} training, {self.X_test.shape[0]} test samples")
        
    def train_random_forest(self):
        """Train Random Forest model with hyperparameter tuning."""
        print("\n🌲 Training Random Forest...")
        
        rf = RandomForestClassifier(random_state=RANDOM_STATE)
        rf_param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf_search = RandomizedSearchCV(
            rf, param_distributions=rf_param_dist,
            n_iter=20, cv=CV_FOLDS, n_jobs=-1, verbose=1,
            random_state=RANDOM_STATE, scoring='f1'
        )
        rf_search.fit(self.X_train, self.y_train)
        
        self.models['random_forest'] = rf_search.best_estimator_
        self.results['random_forest'] = evaluate_model(
            rf_search.best_estimator_, self.X_test, self.y_test, "Random Forest"
        )
        
        return rf_search.best_estimator_
    
    def train_xgboost(self):
        """Train XGBoost model with hyperparameter tuning."""
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost not available. Install with: pip install xgboost")
            return None
            
        print("\n🚀 Training XGBoost...")
        
        xgb_model = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
        xgb_param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        xgb_search = RandomizedSearchCV(
            xgb_model, param_distributions=xgb_param_dist,
            n_iter=20, cv=CV_FOLDS, n_jobs=-1, verbose=1,
            random_state=RANDOM_STATE, scoring='f1'
        )
        xgb_search.fit(self.X_train, self.y_train)
        
        self.models['xgboost'] = xgb_search.best_estimator_
        self.results['xgboost'] = evaluate_model(
            xgb_search.best_estimator_, self.X_test, self.y_test, "XGBoost"
        )
        
        return xgb_search.best_estimator_
    
    def train_lightgbm(self):
        """Train LightGBM model with hyperparameter tuning."""
        if not LIGHTGBM_AVAILABLE:
            print("❌ LightGBM not available. Install with: pip install lightgbm")
            return None
            
        print("\n💡 Training LightGBM...")
        
        lgb_model = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
        lgb_param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'num_leaves': [31, 62, 127]
        }
        
        lgb_search = RandomizedSearchCV(
            lgb_model, param_distributions=lgb_param_dist,
            n_iter=20, cv=CV_FOLDS, n_jobs=-1, verbose=1,
            random_state=RANDOM_STATE, scoring='f1'
        )
        lgb_search.fit(self.X_train, self.y_train)
        
        self.models['lightgbm'] = lgb_search.best_estimator_
        self.results['lightgbm'] = evaluate_model(
            lgb_search.best_estimator_, self.X_test, self.y_test, "LightGBM"
        )
        
        return lgb_search.best_estimator_
    
    def create_ensemble(self):
        """Create ensemble models."""
        print("\n🎯 Creating Ensemble Models...")
        
        # Collect available models
        available_models = []
        for name, model in self.models.items():
            available_models.append((name, model))
        
        if len(available_models) < 2:
            print("⚠️ Need at least 2 models for ensemble. Training individual models first.")
            return None
        
        # Voting Classifier (Hard Voting)
        voting_hard = VotingClassifier(estimators=available_models, voting='hard')
        voting_hard.fit(self.X_train, self.y_train)
        self.models['voting_hard'] = voting_hard
        self.results['voting_hard'] = evaluate_model(
            voting_hard, self.X_test, self.y_test, "Voting Classifier (Hard)"
        )
        
        # Voting Classifier (Soft Voting)
        voting_soft = VotingClassifier(estimators=available_models, voting='soft')
        voting_soft.fit(self.X_train, self.y_train)
        self.models['voting_soft'] = voting_soft
        self.results['voting_soft'] = evaluate_model(
            voting_soft, self.X_test, self.y_test, "Voting Classifier (Soft)"
        )
        
        return voting_soft
    
    def get_best_model(self):
        """Get the best performing model based on F1 score."""
        if not self.results:
            return None
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1'])
        best_model = self.models[best_model_name]
        best_score = self.results[best_model_name]['f1']
        
        print(f"\n🥇 Best Model: {best_model_name}")
        print(f"Best F1-Score: {best_score:.4f}")
        
        return best_model, best_model_name, best_score
    
    def save_models(self, output_dir="models"):
        """Save all trained models."""
        print(f"\n💾 Saving models to {output_dir}...")
        
        for name, model in self.models.items():
            model_path = f"{output_dir}/{name}_model.pkl"
            save_model(model, model_path)
        
        # Save feature names
        joblib.dump(FEATURES, f"{output_dir}/feature_names.pkl")
        
        # Save model metadata
        metadata = {
            'best_model': self.get_best_model()[1] if self.get_best_model() else None,
            'best_f1_score': self.get_best_model()[2] if self.get_best_model() else None,
            'models_available': list(self.models.keys()),
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_names': FEATURES
        }
        joblib.dump(metadata, f"{output_dir}/model_metadata.pkl")
        
        print("✅ All models saved successfully!")
    
    def print_comparison(self):
        """Print model comparison table."""
        if not self.results:
            print("No models to compare.")
            return
        
        print("\n🏆 Model Comparison Summary:")
        print("-" * 60)
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 60)
        
        for name, results in self.results.items():
            print(f"{name:<25} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                  f"{results['recall']:<10.4f} {results['f1']:<10.4f}")
    
    def train_all_models(self):
        """Train all available models."""
        print("🚚 Starting comprehensive model training...")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Train individual models
        self.train_random_forest()
        
        if XGBOOST_AVAILABLE:
            self.train_xgboost()
        
        if LIGHTGBM_AVAILABLE:
            self.train_lightgbm()
        
        # Create ensemble
        self.create_ensemble()
        
        # Print comparison
        self.print_comparison()
        
        # Get best model
        best_model, best_name, best_score = self.get_best_model()
        
        print(f"\n🎉 Training completed!")
        print(f"Best model: {best_name} with F1-score: {best_score:.4f}")
        
        return best_model

def main():
    """Main training function."""
    trainer = ModelTrainer()
    best_model = trainer.train_all_models()
    trainer.save_models()
    
    return best_model

if __name__ == "__main__":
    main() 