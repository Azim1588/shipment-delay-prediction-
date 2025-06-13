import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

print("🚚 Starting Shipment Delay Model Training...")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest + Hyperparameter Tuning
print("🌲 Training Random Forest model...")
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

print("🔍 Performing hyperparameter tuning...")
search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, verbose=1, random_state=42)
search.fit(X_train, y_train)
best_rf = search.best_estimator_

# Evaluation
print("📈 Evaluating model...")
y_pred = best_rf.predict(X_test)

print("Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")

# Save the model
print("💾 Saving model...")
joblib.dump(best_rf, 'random_forest_delay_model.pkl')
print("✅ Model saved successfully as 'random_forest_delay_model.pkl'")

# Print feature importance
print("\n🔝 Feature Importance:")
importances = best_rf.feature_importances_
sorted_idx = np.argsort(importances)
for i, idx in enumerate(sorted_idx):
    print(f"{i+1}. {features[idx]}: {importances[idx]:.4f}")

print("\n🎉 Model training completed! You can now run the Streamlit app.") 