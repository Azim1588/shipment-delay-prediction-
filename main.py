"""
Main entry point for the Shipment Delay Predictor project.
"""
import sys
import os
from pathlib import Path
import io

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st
from src.config.settings import STREAMLIT_CONFIG
from src.utils.model_utils import load_model, predict_with_proper_features
from src.utils.data_loader import load_shipment_data, preprocess_data, get_data_summary
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(**STREAMLIT_CONFIG)

def main():
    """Main application entry point."""
    st.title("🚚 Shipment Delay Predictor - Main Hub")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    app_choice = st.sidebar.selectbox(
        "Choose Application",
        [
            "🏠 Main Dashboard",
            "🔮 Prediction App",
            "📊 Analytics Dashboard", 
            "📋 Custom Reports",
            "🤖 Model Training",
            "ℹ️ Project Info"
        ]
    )
    
    if app_choice == "🏠 Main Dashboard":
        show_main_dashboard()
    elif app_choice == "🔮 Prediction App":
        show_prediction_app()
    elif app_choice == "📊 Analytics Dashboard":
        show_analytics_dashboard()
    elif app_choice == "📋 Custom Reports":
        show_custom_reports()
    elif app_choice == "🤖 Model Training":
        show_model_training()
    elif app_choice == "ℹ️ Project Info":
        show_project_info()

def show_main_dashboard():
    """Show main dashboard with project overview."""
    st.header("🏠 Main Dashboard")
    
    # Project overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Project Overview")
        st.write("""
        **Shipment Delay Predictor** is a comprehensive machine learning solution that:
        
        ✅ **Predicts shipment delays** using ensemble methods
        ✅ **Provides real-time analytics** and insights
        ✅ **Generates custom reports** with export capabilities
        ✅ **Offers automated email notifications**
        ✅ **Supports batch processing** for large datasets
        """)
    
    with col2:
        st.subheader("🚀 Quick Actions")
        
        st.info("💡 **Tip**: Use the sidebar navigation to access different features!")
        
        # Display available features
        st.write("**Available Features:**")
        st.write("• 🔮 **Prediction App** - Make individual predictions")
        st.write("• 📊 **Analytics Dashboard** - View performance metrics")
        st.write("• 📋 **Custom Reports** - Generate and export reports")
        st.write("• 🤖 **Model Training** - Train new ensemble models")
        st.write("• ℹ️ **Project Info** - View documentation and structure")
    
    # System status
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Check if data file exists
        data_exists = os.path.exists("data/DataCoSupplyChainDataset.csv")
        if data_exists:
            st.success("✅ Data File")
        else:
            st.error("❌ Data File")
    
    with col2:
        # Check if model file exists
        model_exists = (os.path.exists("models/random_forest_model.pkl") or 
                       os.path.exists("random_forest_delay_model.pkl"))
        if model_exists:
            st.success("✅ Model Files")
        else:
            st.warning("⚠️ Model Files")
    
    with col3:
        # Check if requirements are installed
        try:
            import streamlit, pandas, numpy, sklearn, plotly
            st.success("✅ Dependencies")
        except ImportError:
            st.error("❌ Dependencies")
    
    with col4:
        # Check if email config exists
        email_config_exists = os.path.exists(".streamlit/secrets.toml")
        if email_config_exists:
            st.success("✅ Email Config")
        else:
            st.warning("⚠️ Email Config")

def show_prediction_app():
    """Show prediction application."""
    st.header("🔮 Shipment Delay Prediction")
    
    # Load model
    try:
        model_paths = [
            "models/ensemble_model.pkl",
            "models/random_forest_model.pkl", 
            "random_forest_delay_model.pkl"
        ]
        
        model = None
        for path in model_paths:
            if os.path.exists(path):
                model = load_model(path)
                break
        
        if model is None:
            st.error("❌ No model found. Please train a model first.")
            st.info("💡 **Solution**: Go to '🤖 Model Training' to train a new model.")
            return
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # Prediction form
    st.subheader("Enter Shipment Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
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
            value=5
        )
    
    with col2:
        quantity = st.slider(
            "Order Item Quantity",
            min_value=1,
            max_value=100,
            value=10
        )
        
        discount_rate = st.slider(
            "Order Item Discount Rate",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.01
        )
        
        profit_ratio = st.slider(
            "Order Item Profit Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01
        )
        
        sales = st.number_input(
            "Sales Amount ($)",
            min_value=0.0,
            value=200.0,
            step=10.0
        )
        
        item_total = st.number_input(
            "Order Item Total ($)",
            min_value=0.0,
            value=250.0,
            step=10.0
        )
    
    # Make prediction
    if st.button("🔮 Predict Delay", type="primary", use_container_width=True):
        features = [
            shipping_mode, customer_segment, order_region, order_state,
            days_scheduled, quantity, discount_rate, profit_ratio, sales, item_total
        ]
        
        try:
            predictions, probabilities = predict_with_proper_features(model, features)
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            if predictions[0] == 1:
                st.error("⏱️ **LIKELY DELAYED**")
                delay_prob = probabilities[0][1] * 100
                st.metric("Delay Probability", f"{delay_prob:.1f}%")
            else:
                st.success("✅ **LIKELY ON TIME**")
                on_time_prob = probabilities[0][0] * 100
                st.metric("On-Time Probability", f"{on_time_prob:.1f}%")
            
            # Probability chart
            fig = go.Figure(data=[
                go.Bar(
                    x=['On Time', 'Delayed'],
                    y=[probabilities[0][0], probabilities[0][1]],
                    marker_color=['green', 'red']
                )
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

def show_analytics_dashboard():
    """Show analytics dashboard."""
    st.header("📊 Analytics Dashboard")
    
    st.info("📊 **Analytics Dashboard** - Real-time performance metrics and insights")
    
    # Load data for analytics
    try:
        df = load_shipment_data("data/DataCoSupplyChainDataset.csv")
        df = preprocess_data(df)
        summary = get_data_summary(df)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{summary['total_records']:,}")
        
        with col2:
            st.metric("Delay Rate", f"{summary['delay_rate']:.1f}%")
        
        with col3:
            st.metric("Total Sales", f"${summary['total_sales']:,.0f}")
        
        with col4:
            st.metric("Avg Sales", f"${summary['avg_sales']:.2f}")
        
        # Analytics charts
        st.markdown("---")
        st.subheader("📈 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Delay rate by customer segment
            segment_delays = df.groupby('Customer Segment')['delayed'].mean().reset_index()
            segment_delays['Customer Segment'] = segment_delays['Customer Segment'].map({
                0: 'Consumer', 1: 'Corporate', 2: 'Home Office', 3: 'Small Business'
            })
            
            fig = px.bar(segment_delays, x='Customer Segment', y='delayed', 
                        title="Delay Rate by Customer Segment")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Shipping mode performance
            shipping_delays = df.groupby('Shipping Mode')['delayed'].mean().reset_index()
            shipping_delays['Shipping Mode'] = shipping_delays['Shipping Mode'].map({
                0: 'Standard', 1: 'Express', 2: 'First Class', 3: 'Same Day'
            })
            
            fig = px.pie(shipping_delays, values='delayed', names='Shipping Mode',
                        title="Delay Distribution by Shipping Mode")
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly trends
        if 'order_month' in df.columns:
            st.subheader("📅 Monthly Trends")
            monthly_data = df.groupby('order_month').agg({
                'delayed': ['count', 'sum', 'mean'],
                'Sales': 'sum'
            }).reset_index()
            monthly_data.columns = ['Month', 'Total_Shipments', 'Delayed_Shipments', 'Delay_Rate', 'Total_Sales']
            monthly_data['Month'] = monthly_data['Month'].astype(str)
            
            fig = px.line(monthly_data, x='Month', y='Delay_Rate', 
                         title="Monthly Delay Rate Trend")
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error loading analytics data: {str(e)}")
        st.info("💡 **Solution**: Ensure the data file is in the 'data/' directory.")

def show_custom_reports():
    """Show custom reports with full functionality."""
    st.header("📋 Custom Reports & Analytics")
    st.markdown("---")
    
    # Load data
    with st.spinner("🔄 Loading data..."):
        df = load_custom_reports_data()
    
    if df is None:
        st.error("❌ Unable to load data. Please ensure DataCoSupplyChainDataset.csv is available.")
        st.info("""
        **Troubleshooting:**
        1. Check if the dataset file exists in the `data/` folder
        2. Ensure the file is named `DataCoSupplyChainDataset.csv`
        3. Verify the file is not corrupted
        """)
        return
    
    st.success(f"✅ Data loaded successfully: {len(df):,} records")
    
    # Sidebar for report configuration
    st.sidebar.title("📊 Report Configuration")
    
    # Report type selection
    report_type = st.sidebar.selectbox(
        "Report Type",
        ["Custom Analytics", "Performance Summary", "Trend Analysis", "Risk Assessment", "Cost Analysis"]
    )
    
    # Date range filter
    try:
        min_date = df['order date (DateOrders)'].min()
        max_date = df['order date (DateOrders)'].max()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Apply date filter
        if len(date_range) == 2:
            mask = (
                (df['order date (DateOrders)'].dt.date >= date_range[0]) &
                (df['order date (DateOrders)'].dt.date <= date_range[1])
            )
            filtered_df = df[mask]
            st.sidebar.success(f"📅 Filtered to {len(filtered_df):,} records")
        else:
            filtered_df = df
            st.sidebar.warning("⚠️ Please select a date range")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error with date filtering: {str(e)}")
        filtered_df = df
    
    # Main content based on report type
    if report_type == "Custom Analytics":
        show_custom_analytics_section(filtered_df)
    elif report_type == "Performance Summary":
        show_performance_summary_section(filtered_df)
    elif report_type == "Trend Analysis":
        show_trend_analysis_section(filtered_df)
    elif report_type == "Risk Assessment":
        show_risk_assessment_section(filtered_df)
    elif report_type == "Cost Analysis":
        show_cost_analysis_section(filtered_df)

@st.cache_data
def load_custom_reports_data():
    """Load and prepare data for custom reports"""
    try:
        # Try multiple possible paths for the dataset
        possible_paths = [
            "DataCoSupplyChainDataset.csv",
            "data/DataCoSupplyChainDataset.csv",
            "src/data/DataCoSupplyChainDataset.csv"
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path, encoding='ISO-8859-1')
                st.success(f"✅ Data loaded from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            st.error("❌ Could not find DataCoSupplyChainDataset.csv in any expected location")
            st.info("Please ensure the dataset file is in the root directory or data/ folder")
            return None
            
        # Data preprocessing
        df['delayed'] = (df['Days for shipping (real)'] > df['Days for shipment (scheduled)']).astype(int)
        df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
        df['order_month'] = df['order date (DateOrders)'].dt.to_period('M')
        df['order_week'] = df['order date (DateOrders)'].dt.to_period('W')
        
        # Encode categorical for analysis
        cat_cols = ['Shipping Mode', 'Customer Segment', 'Order Region', 'Order State']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes
            
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Please check if the dataset file exists and is accessible")
        return None

def generate_custom_chart(df, chart_type, x_column, y_column, color_column=None, title="Custom Chart"):
    """Generate custom charts based on user selection"""
    try:
        # Validate columns exist
        if x_column not in df.columns:
            st.error(f"❌ Column '{x_column}' not found in dataset")
            return None
            
        if y_column not in df.columns:
            st.error(f"❌ Column '{y_column}' not found in dataset")
            return None
            
        if color_column and color_column not in df.columns:
            st.error(f"❌ Color column '{color_column}' not found in dataset")
            return None
        
        # Handle missing values
        df_clean = df.dropna(subset=[x_column, y_column])
        if len(df_clean) == 0:
            st.error("❌ No data available after removing missing values")
            return None
            
        if chart_type == "Bar Chart":
            fig = px.bar(df_clean, x=x_column, y=y_column, color=color_column, title=title)
        elif chart_type == "Line Chart":
            fig = px.line(df_clean, x=x_column, y=y_column, color=color_column, title=title)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df_clean, x=x_column, y=y_column, color=color_column, title=title)
        elif chart_type == "Pie Chart":
            # For pie chart, aggregate the data
            if pd.api.types.is_numeric_dtype(df_clean[y_column]):
                pie_data = df_clean.groupby(x_column)[y_column].sum().reset_index()
                fig = px.pie(pie_data, values=y_column, names=x_column, title=title)
            else:
                pie_data = df_clean[x_column].value_counts().reset_index()
                pie_data.columns = [x_column, 'count']
                fig = px.pie(pie_data, values='count', names=x_column, title=title)
        elif chart_type == "Histogram":
            fig = px.histogram(df_clean, x=x_column, title=title)
        elif chart_type == "Box Plot":
            fig = px.box(df_clean, x=x_column, y=y_column, title=title)
        else:
            fig = px.bar(df_clean, x=x_column, y=y_column, title=title)
        
        fig.update_layout(height=500)
        return fig
    except Exception as e:
        st.error(f"❌ Error generating chart: {str(e)}")
        st.info("Please check your column selections and data types")
        return None

def show_custom_analytics_section(df):
    """Custom analytics with user-defined charts"""
    st.header("🎨 Custom Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Chart Configuration")
        
        # Chart type selection
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"]
        )
        
        # Column selection
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        x_column = st.selectbox("X-Axis Column", df.columns.tolist())
        y_column = st.selectbox("Y-Axis Column", numeric_columns)
        
        if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
            color_column = st.selectbox("Color Column (Optional)", ["None"] + categorical_columns)
            if color_column == "None":
                color_column = None
        else:
            color_column = None
        
        chart_title = st.text_input("Chart Title", f"{chart_type} - {x_column} vs {y_column}")
        
        # Generate chart
        if st.button("Generate Chart"):
            fig = generate_custom_chart(df, chart_type, x_column, y_column, color_column, chart_title)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Data Summary")
        
        # Summary statistics
        summary_stats = {
            "Total Records": len(df),
            "Delay Rate": f"{df['delayed'].mean()*100:.2f}%",
            "Average Sales": f"${df['Sales'].mean():.2f}",
            "Total Sales": f"${df['Sales'].sum():,.2f}",
            "Unique Customers": df['Customer Segment'].nunique(),
            "Date Range": f"{df['order date (DateOrders)'].min().strftime('%Y-%m-%d')} to {df['order date (DateOrders)'].max().strftime('%Y-%m-%d')}"
        }
        
        for key, value in summary_stats.items():
            st.metric(key, value)
    
    # Export options
    st.markdown("---")
    st.subheader("📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export to Excel"):
            # Create Excel report
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Summary sheet
                summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            output.seek(0)
            st.download_button(
                label="📥 Download Excel File",
                data=output.getvalue(),
                file_name=f"custom_analytics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📄 Generate PDF Report"):
            st.info("📄 PDF generation requires additional setup. Please use the standalone custom reports for full PDF functionality.")

def show_performance_summary_section(df):
    """Performance summary report"""
    st.header("📈 Performance Summary Report")
    
    # Calculate performance metrics
    total_shipments = len(df)
    delay_rate = df['delayed'].mean() * 100
    avg_delay_days = df[df['delayed'] == 1]['Days for shipping (real)'].mean()
    total_sales = df['Sales'].sum()
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Shipments", f"{total_shipments:,}")
    with col2:
        st.metric("Delay Rate", f"{delay_rate:.1f}%")
    with col3:
        st.metric("Avg Delay Days", f"{avg_delay_days:.1f}")
    with col4:
        st.metric("Total Sales", f"${total_sales:,.0f}")
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer segment performance
        segment_delays = df.groupby('Customer Segment')['delayed'].mean().reset_index()
        segment_delays['Customer Segment'] = segment_delays['Customer Segment'].map({
            0: 'Consumer', 1: 'Corporate', 2: 'Home Office', 3: 'Small Business'
        })
        
        fig = px.bar(segment_delays, x='Customer Segment', y='delayed',
                    title="Delay Rate by Customer Segment")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Shipping mode performance
        shipping_delays = df.groupby('Shipping Mode')['delayed'].mean().reset_index()
        shipping_delays['Shipping Mode'] = shipping_delays['Shipping Mode'].map({
            0: 'Standard', 1: 'Express', 2: 'First Class', 3: 'Same Day'
        })
        
        fig = px.pie(shipping_delays, values='delayed', names='Shipping Mode',
                    title="Delay Distribution by Shipping Mode")
        st.plotly_chart(fig, use_container_width=True)

def show_trend_analysis_section(df):
    """Trend analysis report"""
    st.header("📊 Trend Analysis")
    
    # Monthly trends
    if 'order_month' in df.columns:
        monthly_data = df.groupby('order_month').agg({
            'delayed': ['count', 'sum', 'mean'],
            'Sales': 'sum'
        }).reset_index()
        monthly_data.columns = ['Month', 'Total_Shipments', 'Delayed_Shipments', 'Delay_Rate', 'Total_Sales']
        monthly_data['Month'] = monthly_data['Month'].astype(str)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(monthly_data, x='Month', y='Delay_Rate', 
                         title="Monthly Delay Rate Trend")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(monthly_data, x='Month', y='Total_Sales', 
                         title="Monthly Sales Trend")
            st.plotly_chart(fig, use_container_width=True)

def show_risk_assessment_section(df):
    """Risk assessment report"""
    st.header("⚠️ Risk Assessment")
    
    # Calculate risk scores
    risk_factors = {
        'High_Quantity': df['Order Item Quantity'] > df['Order Item Quantity'].quantile(0.8),
        'High_Value': df['Sales'] > df['Sales'].quantile(0.8),
        'Long_Distance': df['Order Region'] != df['Order State'],
        'Express_Shipping': df['Shipping Mode'] == 1
    }
    
    df['risk_score'] = sum(risk_factors.values())
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='risk_score', title="Risk Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        risk_summary = df['risk_score'].value_counts().sort_index()
        fig = px.bar(x=risk_summary.index, y=risk_summary.values, 
                    title="Risk Score Breakdown")
        st.plotly_chart(fig, use_container_width=True)

def show_cost_analysis_section(df):
    """Cost analysis report"""
    st.header("💰 Cost Analysis")
    
    # Calculate costs
    total_shipping_cost = df['Sales'].sum() * 0.1  # Assume 10% shipping cost
    total_delay_penalty = df[df['delayed'] == 1]['Sales'].sum() * 0.05  # 5% penalty for delays
    total_cost = total_shipping_cost + total_delay_penalty
    potential_savings = total_delay_penalty * 0.5  # 50% of penalties could be saved
    
    # Cost metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Shipping Cost", f"${total_shipping_cost:,.0f}")
    with col2:
        st.metric("Delay Penalties", f"${total_delay_penalty:,.0f}")
    with col3:
        st.metric("Total Cost", f"${total_cost:,.0f}")
    with col4:
        st.metric("Potential Savings", f"${potential_savings:,.0f}")
    
    # Cost breakdown chart
    cost_data = pd.DataFrame({
        'Category': ['Shipping Cost', 'Delay Penalties'],
        'Amount': [total_shipping_cost, total_delay_penalty]
    })
    
    fig = px.pie(cost_data, values='Amount', names='Category', title="Cost Breakdown")
    st.plotly_chart(fig, use_container_width=True)

def show_model_training():
    """Show model training interface."""
    st.header("🤖 Model Training")
    
    st.subheader("🚀 Train New Ensemble Models")
    st.write("""
    This will train comprehensive ensemble models including:
    - **Random Forest** with hyperparameter tuning
    - **XGBoost** (if available) with advanced gradient boosting
    - **LightGBM** (if available) for fast training
    - **Ensemble voting classifiers** for optimal performance
    """)
    
    # Training options
    st.subheader("⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Types:**")
        st.write("• Random Forest Classifier")
        st.write("• XGBoost (if installed)")
        st.write("• LightGBM (if installed)")
        st.write("• Voting Classifiers")
    
    with col2:
        st.write("**Features:**")
        st.write("• Hyperparameter tuning")
        st.write("• Cross-validation")
        st.write("• Model comparison")
        st.write("• Automatic saving")
    
    st.markdown("---")
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("Training models... This may take several minutes."):
            try:
                from src.models.trainer import ModelTrainer
                trainer = ModelTrainer()
                best_model = trainer.train_all_models()
                trainer.save_models("models")
                
                st.success("✅ Model training completed!")
                
                # Show results
                if trainer.get_best_model():
                    best_model, best_name, best_score = trainer.get_best_model()
                    st.write(f"**Best Model:** {best_name}")
                    st.write(f"**Best F1-Score:** {best_score:.4f}")
                
                # Show comparison table
                if trainer.results:
                    st.subheader("📊 Model Comparison")
                    comparison_data = []
                    for name, results in trainer.results.items():
                        comparison_data.append({
                            'Model': name,
                            'Accuracy': f"{results['accuracy']:.4f}",
                            'Precision': f"{results['precision']:.4f}",
                            'Recall': f"{results['recall']:.4f}",
                            'F1-Score': f"{results['f1']:.4f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")
                st.info("💡 **Solution**: Check that all dependencies are installed and data file is available.")

def show_project_info():
    """Show project information."""
    st.header("ℹ️ Project Information")
    
    st.subheader("📁 Project Structure")
    st.write("""
    ```
    shipment_delay_predictor/
    ├── 📁 src/                          # Source code
    │   ├── 📁 apps/                     # Streamlit applications
    │   ├── 📁 config/                   # Configuration settings
    │   ├── 📁 models/                   # Model training modules
    │   ├── 📁 utils/                    # Utility functions
    │   ├── 📁 data/                     # Data processing
    │   └── 📁 reports/                  # Report generation
    ├── 📁 models/                       # Trained model files
    ├── 📁 data/                         # Data files
    ├── 📁 reports/                      # Generated reports
    ├── 📄 main.py                       # Main entry point
    ├── 📄 requirements.txt              # Dependencies
    └── 📄 README.md                     # Documentation
    ```
    """)
    
    st.subheader("🔧 Features")
    st.write("""
    ✅ **Machine Learning**: Ensemble methods with Random Forest, XGBoost, LightGBM
    ✅ **Web Applications**: Streamlit-based user interfaces
    ✅ **Analytics**: Real-time dashboards and visualizations
    ✅ **Reporting**: Custom reports with PDF/Excel export
    ✅ **Automation**: Email notifications and batch processing
    ✅ **Modular Design**: Clean, maintainable code structure
    """)
    
    st.subheader("📊 Performance")
    st.write("""
    - **Accuracy**: 70%+ (ensemble models)
    - **Precision**: 87%+ for delay detection
    - **Recall**: 56%+ for catching delays
    - **F1-Score**: 68%+ balanced performance
    """)
    
    st.subheader("🚀 Getting Started")
    st.write("""
    1. **Install Dependencies**: `pip install -r requirements.txt`
    2. **Run Main App**: `streamlit run main.py`
    3. **Train Models**: Use the Model Training section
    4. **Make Predictions**: Use the Prediction App
    5. **View Analytics**: Explore the Analytics Dashboard
    6. **Generate Reports**: Use the Custom Reports section
    """)

if __name__ == "__main__":
    main() 