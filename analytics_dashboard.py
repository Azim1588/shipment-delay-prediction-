import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="📊 Shipment Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_sample_data():
    """Load and prepare sample data for analytics"""
    try:
        df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='ISO-8859-1')
        
        # Basic preprocessing
        df['delayed'] = (df['Days for shipping (real)'] > df['Days for shipment (scheduled)']).astype(int)
        df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
        df['order_month'] = df['order date (DateOrders)'].dt.to_period('M')
        df['order_week'] = df['order date (DateOrders)'].dt.to_period('W')
        
        # Encode categorical for analysis
        cat_cols = ['Shipping Mode', 'Customer Segment', 'Order Region', 'Order State']
        for col in cat_cols:
            df[col] = df[col].astype('category').cat.codes
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_risk_score(delay_probability, confidence=0.95):
    """Calculate risk score based on delay probability"""
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
    """Calculate potential cost impact of delays"""
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

def main():
    st.title("📊 Shipment Analytics Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_sample_data()
    if df is None:
        st.error("Unable to load data. Please ensure DataCoSupplyChainDataset.csv is available.")
        return
    
    # Sidebar filters
    st.sidebar.title("📊 Dashboard Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=[df['order date (DateOrders)'].min(), df['order date (DateOrders)'].max()],
        min_value=df['order date (DateOrders)'].min(),
        max_value=df['order date (DateOrders)'].max()
    )
    
    # Customer segment filter
    customer_segments = st.sidebar.multiselect(
        "Customer Segments",
        options=df['Customer Segment'].unique(),
        default=df['Customer Segment'].unique()
    )
    
    # Shipping mode filter
    shipping_modes = st.sidebar.multiselect(
        "Shipping Modes",
        options=df['Shipping Mode'].unique(),
        default=df['Shipping Mode'].unique()
    )
    
    # Apply filters
    mask = (
        (df['order date (DateOrders)'].dt.date >= date_range[0]) &
        (df['order date (DateOrders)'].dt.date <= date_range[1]) &
        (df['Customer Segment'].isin(customer_segments)) &
        (df['Shipping Mode'].isin(shipping_modes))
    )
    filtered_df = df[mask]
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_shipments = len(filtered_df)
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid #20c997;">
            <h3 style="color: #20c997; margin: 0;">{total_shipments:,}</h3>
            <p style="margin: 5px 0 0 0; font-weight: bold;">Total Shipments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        delay_rate = filtered_df['delayed'].mean() * 100
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid #dc3545;">
            <h3 style="color: #dc3545; margin: 0;">{delay_rate:.1f}%</h3>
            <p style="margin: 5px 0 0 0; font-weight: bold;">Delay Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_sales = filtered_df['Sales'].mean()
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid #20c997;">
            <h3 style="color: #20c997; margin: 0;">${avg_sales:,.0f}</h3>
            <p style="margin: 5px 0 0 0; font-weight: bold;">Avg Sales</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_sales = filtered_df['Sales'].sum()
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid #20c997;">
            <h3 style="color: #20c997; margin: 0;">${total_sales:,.0f}</h3>
            <p style="margin: 5px 0 0 0; font-weight: bold;">Total Sales</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts section
    st.markdown("---")
    
    # Row 1: Trend Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Monthly Delay Trends")
        
        monthly_delays = filtered_df.groupby('order_month')['delayed'].agg(['count', 'sum', 'mean']).reset_index()
        monthly_delays['order_month'] = monthly_delays['order_month'].astype(str)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_delays['order_month'],
            y=monthly_delays['mean'] * 100,
            mode='lines+markers',
            name='Delay Rate (%)',
            line=dict(color='red', width=3)
        ))
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Delay Rate (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Delay Rate by Customer Segment")
        
        segment_delays = filtered_df.groupby('Customer Segment')['delayed'].mean().reset_index()
        segment_delays['Customer Segment'] = segment_delays['Customer Segment'].map({
            0: 'Consumer', 1: 'Corporate', 2: 'Home Office', 3: 'Small Business'
        })
        
        fig = px.bar(
            segment_delays,
            x='Customer Segment',
            y='delayed',
            title="Delay Rate by Customer Segment",
            color='delayed',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Geographic and Shipping Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Delay Rate by Region")
        
        region_delays = filtered_df.groupby('Order Region')['delayed'].mean().reset_index()
        region_delays['Order Region'] = region_delays['Order Region'].map({
            0: 'North', 1: 'South', 2: 'East', 3: 'West'
        })
        
        fig = px.pie(
            region_delays,
            values='delayed',
            names='Order Region',
            title="Delay Distribution by Region"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚚 Shipping Mode Performance")
        
        shipping_delays = filtered_df.groupby('Shipping Mode')['delayed'].mean().reset_index()
        shipping_delays['Shipping Mode'] = shipping_delays['Shipping Mode'].map({
            0: 'Standard', 1: 'Express', 2: 'First Class', 3: 'Same Day'
        })
        
        fig = px.bar(
            shipping_delays,
            x='Shipping Mode',
            y='delayed',
            title="Delay Rate by Shipping Mode",
            color='delayed',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 3: Risk Analysis and Cost Impact
    st.markdown("---")
    st.subheader("⚠️ Risk Analysis & Cost Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎲 Risk Score Distribution")
        
        # Simulate risk scores for demonstration
        np.random.seed(42)
        sample_size = min(1000, len(filtered_df))
        sample_df = filtered_df.sample(n=sample_size)
        
        # Calculate risk scores
        risk_scores = []
        for _, row in sample_df.iterrows():
            # Simulate delay probability based on features
            base_prob = 0.3
            if row['Shipping Mode'] == 0:  # Standard shipping
                base_prob += 0.1
            if row['Customer Segment'] == 0:  # Consumer
                base_prob += 0.05
            
            risk_score, _, _ = calculate_risk_score(base_prob)
            risk_scores.append(risk_score)
        
        fig = px.histogram(
            x=risk_scores,
            nbins=20,
            title="Distribution of Risk Scores",
            labels={'x': 'Risk Score', 'y': 'Number of Shipments'}
        )
        fig.add_vline(x=40, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
        fig.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="High Risk")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Cost Impact Analysis")
        
        # Calculate cost impact for sample data
        cost_data = []
        for _, row in sample_df.iterrows():
            delay_prob = 0.3  # Simplified for demo
            cost_impact = calculate_cost_impact(delay_prob, row['Sales'])
            cost_data.append(cost_impact)
        
        cost_df = pd.DataFrame(cost_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Shipping Cost', 'Expected Delay Cost', 'Potential Savings'],
            y=[
                cost_df['shipping_cost'].mean(),
                cost_df['expected_delay_cost'].mean(),
                cost_df['cost_savings_potential'].mean()
            ],
            marker_color=['blue', 'red', 'green']
        ))
        fig.update_layout(
            title="Average Cost Breakdown per Shipment",
            yaxis_title="Cost ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 4: Key Insights and Recommendations
    st.markdown("---")
    st.subheader("💡 Key Insights & Recommendations")
    
    # Calculate insights
    insights = []
    
    # Insight 1: Overall performance
    if delay_rate > 50:
        insights.append("🔴 **High Delay Rate**: Consider reviewing shipping processes and carrier performance.")
    elif delay_rate > 30:
        insights.append("🟡 **Moderate Delay Rate**: Monitor trends and identify improvement opportunities.")
    else:
        insights.append("🟢 **Good Performance**: Maintain current processes and continue monitoring.")
    
    # Insight 2: Customer segment analysis
    segment_performance = filtered_df.groupby('Customer Segment')['delayed'].mean()
    worst_segment = segment_performance.idxmax()
    best_segment = segment_performance.idxmin()
    
    segment_names = {0: 'Consumer', 1: 'Corporate', 2: 'Home Office', 3: 'Small Business'}
    insights.append(f"📊 **Segment Performance**: {segment_names[best_segment]} has the best performance, while {segment_names[worst_segment]} needs attention.")
    
    # Insight 3: Shipping mode efficiency
    shipping_performance = filtered_df.groupby('Shipping Mode')['delayed'].mean()
    most_reliable = shipping_performance.idxmin()
    shipping_names = {0: 'Standard', 1: 'Express', 2: 'First Class', 3: 'Same Day'}
    insights.append(f"🚚 **Most Reliable**: {shipping_names[most_reliable]} shipping shows the lowest delay rate.")
    
    # Insight 4: Cost optimization
    avg_sales = filtered_df['Sales'].mean()
    potential_savings = avg_sales * delay_rate / 100 * 0.05  # 5% penalty per delayed shipment
    insights.append(f"💰 **Cost Impact**: Potential savings of ${potential_savings:.2f} per shipment by reducing delays.")
    
    # Display insights
    for insight in insights:
        st.write(insight)
    
    # Recommendations
    st.subheader("🎯 Actionable Recommendations")
    
    recommendations = [
        "**Immediate Actions**:",
        "• Review shipping processes for high-risk segments",
        "• Optimize carrier selection based on performance data",
        "• Implement proactive monitoring for shipments with >70% delay risk",
        "",
        "**Strategic Improvements**:",
        "• Develop targeted strategies for underperforming customer segments",
        "• Consider premium shipping options for high-value orders",
        "• Establish performance-based carrier partnerships",
        "",
        "**Long-term Optimization**:",
        "• Implement machine learning-based route optimization",
        "• Develop predictive maintenance for shipping infrastructure",
        "• Create customer communication protocols for delay management"
    ]
    
    for rec in recommendations:
        st.write(rec)

if __name__ == "__main__":
    main() 