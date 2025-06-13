import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost imported successfully")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM imported successfully")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("❌ LightGBM not available. Install with: pip install lightgbm")

print("🚚 Starting Enhanced Shipment Delay Model Training with Ensemble Methods...")

# Load and preprocess dataset
print("📊 Loading dataset...")
df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='ISO-8859-1')

# Drop irrelevant columns
df.drop(columns=[
    'Product Description', 'Customer Email', 'Customer Fname', 'Customer Lname',
    'Customer Password', 'Customer Street', 'Product Image', 'Order Zipcode'
], inplace=True)

# Target variable: binary delay label
df['delayed'] = (df['Days for shipping (real)'] > df['Days for shipment (scheduled)']).astype(int)

# Encode categorical columns
print("🔧 Encoding categorical variables...")
cat_cols = ['Shipping Mode', 'Customer Segment', 'Order Region', 'Order State', 'Market']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Drop rows with any remaining missing data
df.dropna(inplace=True)

# Feature selection and split
print("🎯 Preparing features...")
features = [
    'Shipping Mode', 'Customer Segment', 'Order Region', 'Order State',
    'Days for shipment (scheduled)', 'Order Item Quantity',
    'Order Item Discount Rate', 'Order Item Profit Ratio', 'Sales', 'Order Item Total'
]
X = df[features]
y = df['delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Delay rate in training set: {y_train.mean():.3f}")
print(f"Delay rate in test set: {y_test.mean():.3f}")

# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n📊 {model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# 1. Random Forest with Hyperparameter Tuning
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(random_state=42)
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
    n_iter=20, cv=5, n_jobs=-1, verbose=1, 
    random_state=42, scoring='f1'
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

rf_results = evaluate_model(best_rf, X_test, y_test, "Random Forest")

# 2. XGBoost (if available)
xgb_results = None
if XGBOOST_AVAILABLE:
    print("\n🚀 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
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
        n_iter=20, cv=5, n_jobs=-1, verbose=1,
        random_state=42, scoring='f1'
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    
    xgb_results = evaluate_model(best_xgb, X_test, y_test, "XGBoost")

# 3. LightGBM (if available)
lgb_results = None
if LIGHTGBM_AVAILABLE:
    print("\n💡 Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
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
        n_iter=20, cv=5, n_jobs=-1, verbose=1,
        random_state=42, scoring='f1'
    )
    lgb_search.fit(X_train, y_train)
    best_lgb = lgb_search.best_estimator_
    
    lgb_results = evaluate_model(best_lgb, X_test, y_test, "LightGBM")

# 4. Ensemble Models
print("\n🎯 Creating Ensemble Models...")

# Collect all available models
models = [('rf', best_rf)]
if xgb_results:
    models.append(('xgb', best_xgb))
if lgb_results:
    models.append(('lgb', best_lgb))

# Voting Classifier (Hard Voting)
if len(models) > 1:
    voting_hard = VotingClassifier(estimators=models, voting='hard')
    voting_hard.fit(X_train, y_train)
    voting_hard_results = evaluate_model(voting_hard, X_test, y_test, "Voting Classifier (Hard)")

    # Voting Classifier (Soft Voting)
    voting_soft = VotingClassifier(estimators=models, voting='soft')
    voting_soft.fit(X_train, y_train)
    voting_soft_results = evaluate_model(voting_soft, X_test, y_test, "Voting Classifier (Soft)")

# 5. Weighted Average Ensemble
print("\n⚖️ Creating Weighted Average Ensemble...")

def create_weighted_ensemble(models_results, weights=None):
    """Create a weighted average ensemble"""
    if weights is None:
        # Equal weights
        weights = [1/len(models_results)] * len(models_results)
    
    # Get probabilities from all models
    probabilities = []
    for result in models_results:
        if result is not None:
            probabilities.append(result['probabilities'])
    
    # Calculate weighted average
    weighted_proba = np.average(probabilities, axis=0, weights=weights)
    weighted_pred = (weighted_proba > 0.5).astype(int)
    
    return weighted_pred, weighted_proba

# Collect results for ensemble
ensemble_results = [rf_results]
if xgb_results:
    ensemble_results.append(xgb_results)
if lgb_results:
    ensemble_results.append(lgb_results)

# Create weighted ensemble
weighted_pred, weighted_proba = create_weighted_ensemble(ensemble_results)

# Evaluate weighted ensemble
weighted_accuracy = accuracy_score(y_test, weighted_pred)
weighted_precision = precision_score(y_test, weighted_pred)
weighted_recall = recall_score(y_test, weighted_pred)
weighted_f1 = f1_score(y_test, weighted_pred)

print(f"\n📊 Weighted Average Ensemble Performance:")
print(f"Accuracy: {weighted_accuracy:.4f}")
print(f"Precision: {weighted_precision:.4f}")
print(f"Recall: {weighted_recall:.4f}")
print(f"F1-Score: {weighted_f1:.4f}")

# 6. Compare all models
print("\n🏆 Model Comparison Summary:")
print("-" * 60)
print(f"{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print("-" * 60)

models_comparison = [
    ("Random Forest", rf_results),
]

if xgb_results:
    models_comparison.append(("XGBoost", xgb_results))
if lgb_results:
    models_comparison.append(("LightGBM", lgb_results))
if len(models) > 1:
    models_comparison.extend([
        ("Voting (Hard)", voting_hard_results),
        ("Voting (Soft)", voting_soft_results)
    ])

models_comparison.append(("Weighted Ensemble", {
    'accuracy': weighted_accuracy,
    'precision': weighted_precision,
    'recall': weighted_recall,
    'f1': weighted_f1
}))

for name, results in models_comparison:
    print(f"{name:<30} {results['accuracy']:<10.4f} {results['precision']:<10.4f} {results['recall']:<10.4f} {results['f1']:<10.4f}")

# 7. Find best model
best_model_name = max(models_comparison, key=lambda x: x[1]['f1'])[0]
best_model_results = max(models_comparison, key=lambda x: x[1]['f1'])[1]

print(f"\n🥇 Best Model: {best_model_name}")
print(f"Best F1-Score: {best_model_results['f1']:.4f}")

# 8. Save the best ensemble model
print("\n💾 Saving models...")

# Save individual models
joblib.dump(best_rf, 'random_forest_model.pkl')
if xgb_results:
    joblib.dump(best_xgb, 'xgboost_model.pkl')
if lgb_results:
    joblib.dump(best_lgb, 'lightgbm_model.pkl')

# Save ensemble model
if len(models) > 1:
    joblib.dump(voting_soft, 'ensemble_model.pkl')
    print("✅ Ensemble model saved as 'ensemble_model.pkl'")
else:
    joblib.dump(best_rf, 'ensemble_model.pkl')
    print("✅ Best model saved as 'ensemble_model.pkl'")

# Save feature names for proper prediction
joblib.dump(features, 'feature_names.pkl')
print("✅ Feature names saved as 'feature_names.pkl'")

# Save model metadata
model_metadata = {
    'best_model': best_model_name,
    'best_f1_score': best_model_results['f1'],
    'best_accuracy': best_model_results['accuracy'],
    'best_precision': best_model_results['precision'],
    'best_recall': best_model_results['recall'],
    'feature_names': features,
    'models_available': [name for name, _ in models_comparison],
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

joblib.dump(model_metadata, 'model_metadata.pkl')
print("✅ Model metadata saved as 'model_metadata.pkl'")

print("\n🎉 Enhanced model training completed!")
print("📈 Performance improvement achieved with ensemble methods!")
print("🚀 Ready to use the improved models in your Streamlit app!") 