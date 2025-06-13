import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="🚚 Shipment Delay Predictor",
    page_icon="🚚",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_delay_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'random_forest_delay_model.pkl' not found. Please ensure the model is saved in the same directory.")
        return None

# Load model
model = load_model()

# Main app
def main():
    st.title("🚚 Shipment Delay Predictor")
    st.markdown("---")
    
    if model is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Predict Delay", "Batch Prediction", "Model Info"]
    )
    
    if page == "Predict Delay":
        show_prediction_page()
    elif page == "Batch Prediction":
        show_batch_prediction_page()
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
        # Prepare input features in the same order as training
        features = np.array([[
            shipping_mode, customer_segment, order_region, order_state,
            days_scheduled, quantity, discount_rate, profit_ratio, sales, item_total
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
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
            required_features = [
                'Shipping Mode', 'Customer Segment', 'Order Region', 'Order State',
                'Days for shipment (scheduled)', 'Order Item Quantity',
                'Order Item Discount Rate', 'Order Item Profit Ratio', 'Sales', 'Order Item Total'
            ]
            
            missing_cols = [col for col in required_features if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.write("Please ensure your CSV file contains all required columns.")
            else:
                if st.button("🔮 Predict Batch Delays", type="primary"):
                    # Prepare features
                    X_batch = df[required_features].values
                    
                    # Make predictions
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
                    
                    # Results table
                    st.write("Detailed Results:")
                    st.dataframe(df[['Predicted_Delay', 'Delay_Probability', 'OnTime_Probability'] + required_features])
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="shipment_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def show_model_info_page():
    st.header("ℹ️ Model Information")
    
    st.subheader("Model Details")
    st.write("""
    **Model Type:** Random Forest Classifier
    **Purpose:** Predict shipment delays based on order and customer characteristics
    **Features Used:**
    """)
    
    features = [
        'Shipping Mode', 'Customer Segment', 'Order Region', 'Order State',
        'Days for shipment (scheduled)', 'Order Item Quantity',
        'Order Item Discount Rate', 'Order Item Profit Ratio', 'Sales', 'Order Item Total'
    ]
    
    for i, feature in enumerate(features, 1):
        st.write(f"{i}. {feature}")
    
    st.subheader("Model Performance")
    st.write("""
    - **Accuracy:** 70.0%
    - **Precision:** 86.9%
    - **Recall:** 56.0%
    """)
    
    st.subheader("How to Use")
    st.write("""
    1. **Single Prediction:** Use the "Predict Delay" page to predict delays for individual shipments
    2. **Batch Prediction:** Use the "Batch Prediction" page to upload a CSV file with multiple shipments
    3. **Input Requirements:** Ensure all required features are provided with appropriate data types
    """)
    
    st.subheader("Feature Descriptions")
    feature_descriptions = {
        'Shipping Mode': 'Method of shipping (Standard, Express, First Class, Same Day)',
        'Customer Segment': 'Type of customer (Consumer, Corporate, Home Office, Small Business)',
        'Order Region': 'Geographic region of the order',
        'Order State': 'State where the order is placed',
        'Days for shipment (scheduled)': 'Scheduled number of days for shipping',
        'Order Item Quantity': 'Quantity of items in the order',
        'Order Item Discount Rate': 'Discount rate applied to the order',
        'Order Item Profit Ratio': 'Profit ratio for the order',
        'Sales': 'Total sales amount',
        'Order Item Total': 'Total value of order items'
    }
    
    for feature, description in feature_descriptions.items():
        st.write(f"**{feature}:** {description}")

if __name__ == "__main__":
    main() 