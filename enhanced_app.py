import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="🚚 Enhanced Shipment Delay Predictor",
    page_icon="🚚",
    layout="wide"
)

# Load models and metadata
@st.cache_resource
def load_models():
    """Load all available models and metadata"""
    models = {}
    metadata = {}
    
    try:
        # Load ensemble model
        models['ensemble'] = joblib.load("ensemble_model.pkl")
        print("✅ Ensemble model loaded")
    except FileNotFoundError:
        try:
            # Fallback to individual models
            models['random_forest'] = joblib.load("random_forest_model.pkl")
            print("✅ Random Forest model loaded")
        except FileNotFoundError:
            try:
                models['basic'] = joblib.load("random_forest_delay_model.pkl")
                print("✅ Basic model loaded")
            except FileNotFoundError:
                st.error("❌ No model files found. Please run the training script first.")
                return None, None
    
    try:
        # Load feature names
        models['feature_names'] = joblib.load("feature_names.pkl")
        print("✅ Feature names loaded")
    except FileNotFoundError:
        # Fallback feature names
        models['feature_names'] = [
            'Shipping Mode', 'Customer Segment', 'Order Region', 'Order State',
            'Days for shipment (scheduled)', 'Order Item Quantity',
            'Order Item Discount Rate', 'Order Item Profit Ratio', 'Sales', 'Order Item Total'
        ]
        print("⚠️ Using fallback feature names")
    
    try:
        # Load metadata
        metadata = joblib.load("model_metadata.pkl")
        print("✅ Model metadata loaded")
    except FileNotFoundError:
        metadata = {
            'best_model': 'Basic Model',
            'best_f1_score': 0.68,
            'best_accuracy': 0.70,
            'best_precision': 0.87,
            'best_recall': 0.56,
            'feature_names': models['feature_names'],
            'models_available': ['Basic Model'],
            'training_date': 'Unknown'
        }
        print("⚠️ Using default metadata")
    
    return models, metadata

# Load models
models, metadata = load_models()

if models is None:
    st.stop()

# Main app
def main():
    st.title("🚚 Enhanced Shipment Delay Predictor")
    st.markdown("---")
    
    # Display model info
    with st.sidebar:
        st.title("Model Information")
        st.write(f"**Best Model:** {metadata.get('best_model', 'Unknown')}")
        st.write(f"**F1-Score:** {metadata.get('best_f1_score', 0):.3f}")
        st.write(f"**Accuracy:** {metadata.get('best_accuracy', 0):.3f}")
        st.write(f"**Precision:** {metadata.get('best_precision', 0):.3f}")
        st.write(f"**Recall:** {metadata.get('best_recall', 0):.3f}")
        st.write(f"**Trained:** {metadata.get('training_date', 'Unknown')}")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Predict Delay", "Batch Prediction", "Model Comparison", "Model Info"]
    )
    
    if page == "Predict Delay":
        show_prediction_page()
    elif page == "Batch Prediction":
        show_batch_prediction_page()
    elif page == "Model Comparison":
        show_model_comparison_page()
    elif page == "Model Info":
        show_model_info_page()

def show_prediction_page():
    st.header("📊 Single Shipment Prediction")
    st.write("Enter shipment details to predict whether the shipment will be delayed.")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Shipment Details")
        
        # Input fields matching the model features
        shipping_mode = st.selectbox(
            "Shipping Mode",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Standard", "Express", "First Class", "Same Day"][x] if x < 4 else f"Mode {x}"
        )
        
        customer_segment = st.selectbox(
            "Customer Segment",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Consumer", "Corporate", "Home Office", "Small Business"][x] if x < 4 else f"Segment {x}"
        )
        
        order_region = st.selectbox(
            "Order Region",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["North", "South", "East", "West"][x] if x < 4 else f"Region {x}"
        )
        
        order_state = st.selectbox(
            "Order State",
            options=list(range(10)),
            format_func=lambda x: f"State {x}"
        )
        
        days_scheduled = st.slider(
            "Days for Shipment (Scheduled)",
            min_value=1,
            max_value=30,
            value=5,
            help="Number of days scheduled for shipping"
        )
    
    with col2:
        st.subheader("Order Details")
        
        quantity = st.slider(
            "Order Item Quantity",
            min_value=1,
            max_value=100,
            value=10,
            help="Quantity of items in the order"
        )
        
        discount_rate = st.slider(
            "Order Item Discount Rate",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Discount rate applied to the order"
        )
        
        profit_ratio = st.slider(
            "Order Item Profit Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Profit ratio for the order"
        )
        
        sales = st.number_input(
            "Sales Amount ($)",
            min_value=0.0,
            value=200.0,
            step=10.0,
            help="Total sales amount"
        )
        
        item_total = st.number_input(
            "Order Item Total ($)",
            min_value=0.0,
            value=250.0,
            step=10.0,
            help="Total value of order items"
        )
    
    # Prediction button
    if st.button("🔮 Predict Delay", type="primary", use_container_width=True):
        # Prepare input features with proper feature names
        feature_names = models['feature_names']
        features_data = [
            shipping_mode, customer_segment, order_region, order_state,
            days_scheduled, quantity, discount_rate, profit_ratio, sales, item_total
        ]
        
        # Create DataFrame with proper feature names
        features_df = pd.DataFrame([features_data], columns=feature_names)
        
        # Make prediction using ensemble model
        model = models.get('ensemble') or models.get('random_forest') or models.get('basic')
        
        try:
            prediction = model.predict(features_df)[0]
            prediction_proba = model.predict_proba(features_df)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            # Create result display
            if prediction == 1:
                st.error("⏱️ **LIKELY DELAYED**")
                delay_prob = prediction_proba[1] * 100
                st.metric("Delay Probability", f"{delay_prob:.1f}%")
            else:
                st.success("✅ **LIKELY ON TIME**")
                on_time_prob = prediction_proba[0] * 100
                st.metric("On-Time Probability", f"{on_time_prob:.1f}%")
            
            # Show probability breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.metric("On-Time Probability", f"{prediction_proba[0]*100:.1f}%")
            with col2:
                st.metric("Delay Probability", f"{prediction_proba[1]*100:.1f}%")
            
            # Probability bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=['On Time', 'Delayed'],
                    y=[prediction_proba[0], prediction_proba[1]],
                    marker_color=['green', 'red']
                )
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            st.subheader("⚠️ Risk Assessment")
            if prediction_proba[1] > 0.7:
                st.warning("**HIGH RISK** - Strong likelihood of delay")
            elif prediction_proba[1] > 0.5:
                st.info("**MEDIUM RISK** - Moderate chance of delay")
            else:
                st.success("**LOW RISK** - Likely to be on time")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please ensure all input values are valid.")

def show_batch_prediction_page():
    st.header("📁 Batch Prediction")
    st.write("Upload a CSV file with multiple shipments to predict delays in batch.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with the same column structure as the training data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Check if required columns exist
            required_features = models['feature_names']
            missing_cols = [col for col in required_features if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.write("Please ensure your CSV file contains all required columns.")
            else:
                if st.button("🔮 Predict Batch Delays", type="primary"):
                    # Prepare features with proper feature names
                    X_batch = df[required_features]
                    
                    # Make predictions
                    model = models.get('ensemble') or models.get('random_forest') or models.get('basic')
                    predictions = model.predict(X_batch)
                    probabilities = model.predict_proba(X_batch)
                    
                    # Add predictions to dataframe
                    df['Predicted_Delay'] = predictions
                    df['Delay_Probability'] = probabilities[:, 1]
                    df['OnTime_Probability'] = probabilities[:, 0]
                    
                    # Display results
                    st.subheader("📊 Batch Prediction Results")
                    
                    # Summary statistics
                    total_shipments = len(predictions)
                    delayed_count = sum(predictions)
                    on_time_count = total_shipments - delayed_count
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Shipments", total_shipments)
                    with col2:
                        st.metric("Predicted Delays", delayed_count)
                    with col3:
                        st.metric("Predicted On-Time", on_time_count)
                    
                    # Risk distribution chart
                    fig = px.histogram(
                        df, 
                        x='Delay_Probability', 
                        nbins=20,
                        title="Distribution of Delay Probabilities",
                        labels={'Delay_Probability': 'Delay Probability', 'count': 'Number of Shipments'}
                    )
                    fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="50% Threshold")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.write("Detailed Results:")
                    st.dataframe(df[['Predicted_Delay', 'Delay_Probability', 'OnTime_Probability'] + required_features])
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="enhanced_shipment_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def show_model_comparison_page():
    st.header("📈 Model Comparison")
    st.write("Compare the performance of different ensemble models.")
    
    # Display model performance metrics
    st.subheader("🏆 Model Performance Metrics")
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [
            metadata.get('best_accuracy', 0),
            metadata.get('best_precision', 0),
            metadata.get('best_recall', 0),
            metadata.get('best_f1_score', 0)
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create metrics visualization
    fig = go.Figure(data=[
        go.Bar(
            x=metrics_df['Metric'],
            y=metrics_df['Value'],
            marker_color=['blue', 'green', 'orange', 'red']
        )
    ])
    fig.update_layout(
        title="Model Performance Metrics",
        yaxis_title="Score",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metrics table
    st.dataframe(metrics_df, use_container_width=True)
    
    # Model information
    st.subheader("ℹ️ Model Details")
    st.write(f"**Best Model:** {metadata.get('best_model', 'Unknown')}")
    st.write(f"**Available Models:** {', '.join(metadata.get('models_available', ['Unknown']))}")
    st.write(f"**Training Date:** {metadata.get('training_date', 'Unknown')}")
    st.write(f"**Features Used:** {len(metadata.get('feature_names', []))}")

def show_model_info_page():
    st.header("ℹ️ Enhanced Model Information")
    
    st.subheader("🚀 Ensemble Methods Used")
    st.write("""
    This enhanced version uses multiple machine learning algorithms combined through ensemble methods:
    
    **1. Random Forest Classifier**
    - Robust tree-based algorithm
    - Handles non-linear relationships
    - Good for categorical and numerical features
    
    **2. XGBoost (eXtreme Gradient Boosting)**
    - Advanced gradient boosting algorithm
    - Excellent performance on structured data
    - Built-in regularization to prevent overfitting
    
    **3. LightGBM**
    - Light Gradient Boosting Machine
    - Fast training and prediction
    - Memory-efficient implementation
    
    **4. Ensemble Methods**
    - **Voting Classifier**: Combines predictions from multiple models
    - **Weighted Average**: Assigns different weights to each model
    - **Soft Voting**: Uses probability scores for better accuracy
    """)
    
    st.subheader("📊 Performance Improvements")
    st.write(f"""
    **Enhanced Model Performance:**
    - **Accuracy:** {metadata.get('best_accuracy', 0):.1%} (vs 70% baseline)
    - **Precision:** {metadata.get('best_precision', 0):.1%} (vs 87% baseline)
    - **Recall:** {metadata.get('best_recall', 0):.1%} (vs 56% baseline)
    - **F1-Score:** {metadata.get('best_f1_score', 0):.1%} (vs 68% baseline)
    """)
    
    st.subheader("🔧 Technical Features")
    st.write("""
    **Advanced Features:**
    - **Hyperparameter Tuning**: Optimized model parameters using RandomizedSearchCV
    - **Cross-Validation**: 5-fold cross-validation for robust performance
    - **Feature Engineering**: Proper handling of categorical variables
    - **Error Handling**: Graceful fallback to available models
    - **Real-time Predictions**: Fast inference with ensemble models
    """)
    
    st.subheader("🎯 Business Benefits")
    st.write("""
    **Enhanced Capabilities:**
    - **Higher Accuracy**: More reliable predictions
    - **Better Risk Assessment**: Detailed probability scores
    - **Robust Performance**: Less sensitive to data variations
    - **Scalable Architecture**: Handles large datasets efficiently
    - **Production Ready**: Enterprise-grade reliability
    """)

if __name__ == "__main__":
    main() 