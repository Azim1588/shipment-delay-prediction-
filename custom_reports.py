import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="📋 Custom Reports & Analytics",
    page_icon="📋",
    layout="wide"
)

@st.cache_data
def load_data():
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

def create_pdf_report(report_data, report_title="Custom Report"):
    """Create PDF report with custom analytics"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 20))
    
    # Report metadata
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Summary statistics
    if 'summary_stats' in report_data:
        story.append(Paragraph("Summary Statistics", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        summary_data = [['Metric', 'Value']]
        for key, value in report_data['summary_stats'].items():
            summary_data.append([key, str(value)])
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
    
    # Key insights
    if 'insights' in report_data:
        story.append(Paragraph("Key Insights", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        for insight in report_data['insights']:
            story.append(Paragraph(f"• {insight}", styles['Normal']))
            story.append(Spacer(1, 6))
        
        story.append(Spacer(1, 20))
    
    # Recommendations
    if 'recommendations' in report_data:
        story.append(Paragraph("Recommendations", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        for rec in report_data['recommendations']:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 6))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def send_email_report(report_data, recipient_email, subject="Shipment Analytics Report"):
    """Send email with report attachment"""
    try:
        # Email configuration (you'll need to set these up)
        sender_email = st.secrets.get("email_sender", "your-email@gmail.com")
        sender_password = st.secrets.get("email_password", "your-app-password")
        smtp_server = st.secrets.get("smtp_server", "smtp.gmail.com")
        smtp_port = st.secrets.get("smtp_port", 587)
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Email body
        body = f"""
        Dear User,
        
        Please find attached your custom shipment analytics report.
        
        Report Summary:
        - Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - Total shipments analyzed: {report_data.get('total_shipments', 'N/A')}
        - Delay rate: {report_data.get('delay_rate', 'N/A')}
        
        Best regards,
        Shipment Analytics Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF report
        pdf_buffer = create_pdf_report(report_data, subject)
        attachment = MIMEBase('application', 'pdf')
        attachment.set_payload(pdf_buffer.read())
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', 'attachment', filename=f"shipment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        msg.attach(attachment)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

def main():
    st.title("📋 Custom Reports & Analytics")
    st.markdown("---")
    
    # Load data
    with st.spinner("🔄 Loading data..."):
        df = load_data()
    
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
        show_custom_analytics(filtered_df)
    elif report_type == "Performance Summary":
        show_performance_summary(filtered_df)
    elif report_type == "Trend Analysis":
        show_trend_analysis(filtered_df)
    elif report_type == "Risk Assessment":
        show_risk_assessment(filtered_df)
    elif report_type == "Cost Analysis":
        show_cost_analysis(filtered_df)

def show_custom_analytics(df):
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
    
    # Export and email options
    st.markdown("---")
    st.subheader("📤 Export & Share Options")
    
    col1, col2, col3 = st.columns(3)
    
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
                file_name=f"custom_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📄 Generate PDF Report"):
            # Prepare report data
            report_data = {
                'summary_stats': summary_stats,
                'insights': [
                    f"Delay rate is {df['delayed'].mean()*100:.1f}% for the selected period",
                    f"Total sales volume is ${df['Sales'].sum():,.0f}",
                    f"Average order value is ${df['Sales'].mean():.2f}"
                ],
                'recommendations': [
                    "Monitor high-delay periods for pattern identification",
                    "Consider optimizing shipping methods for better performance",
                    "Review customer segments with highest delay rates"
                ],
                'total_shipments': len(df),
                'delay_rate': f"{df['delayed'].mean()*100:.1f}%"
            }
            
            pdf_buffer = create_pdf_report(report_data, "Custom Analytics Report")
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer.getvalue(),
                file_name=f"custom_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
    
    with col3:
        st.subheader("📧 Email Report")
        recipient_email = st.text_input("Recipient Email")
        email_subject = st.text_input("Email Subject", "Custom Shipment Analytics Report")
        
        if st.button("📤 Send Email Report"):
            if recipient_email:
                report_data = {
                    'summary_stats': summary_stats,
                    'insights': [
                        f"Delay rate is {df['delayed'].mean()*100:.1f}% for the selected period",
                        f"Total sales volume is ${df['Sales'].sum():,.0f}",
                        f"Average order value is ${df['Sales'].mean():.2f}"
                    ],
                    'recommendations': [
                        "Monitor high-delay periods for pattern identification",
                        "Consider optimizing shipping methods for better performance",
                        "Review customer segments with highest delay rates"
                    ],
                    'total_shipments': len(df),
                    'delay_rate': f"{df['delayed'].mean()*100:.1f}%"
                }
                
                if send_email_report(report_data, recipient_email, email_subject):
                    st.success("✅ Email sent successfully!")
                else:
                    st.error("❌ Failed to send email. Please check your email configuration.")
            else:
                st.warning("Please enter a recipient email address.")

def show_performance_summary(df):
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
        # Delay rate by segment
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
    
    # Export options
    st.markdown("---")
    export_performance_report(df, total_shipments, delay_rate, avg_delay_days, total_sales)

def show_trend_analysis(df):
    """Trend analysis report"""
    st.header("📊 Trend Analysis Report")
    
    # Monthly trends
    monthly_data = df.groupby('order_month').agg({
        'delayed': ['count', 'sum', 'mean'],
        'Sales': 'sum'
    }).reset_index()
    monthly_data.columns = ['Month', 'Total_Shipments', 'Delayed_Shipments', 'Delay_Rate', 'Total_Sales']
    monthly_data['Month'] = monthly_data['Month'].astype(str)
    
    # Trend charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(monthly_data, x='Month', y='Delay_Rate', 
                     title="Monthly Delay Rate Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(monthly_data, x='Month', y='Total_Sales', 
                     title="Monthly Sales Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.markdown("---")
    export_trend_report(df, monthly_data)

def show_risk_assessment(df):
    """Risk assessment report"""
    st.header("⚠️ Risk Assessment Report")
    
    # Risk scoring
    np.random.seed(42)
    risk_scores = []
    for _, row in df.sample(n=min(1000, len(df))).iterrows():
        base_prob = 0.3
        if row['Shipping Mode'] == 0:  # Standard shipping
            base_prob += 0.1
        if row['Customer Segment'] == 0:  # Consumer
            base_prob += 0.05
        risk_scores.append(base_prob * 100)
    
    # Risk distribution
    fig = px.histogram(x=risk_scores, nbins=20, 
                      title="Risk Score Distribution",
                      labels={'x': 'Risk Score', 'y': 'Number of Shipments'})
    fig.add_vline(x=40, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
    fig.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="High Risk")
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.markdown("---")
    export_risk_report(df, risk_scores)

def show_cost_analysis(df):
    """Cost analysis report"""
    st.header("💰 Cost Analysis Report")
    
    # Cost calculations
    shipping_cost_ratio = 0.1
    delay_penalty_ratio = 0.05
    
    df['shipping_cost'] = df['Sales'] * shipping_cost_ratio
    df['delay_penalty'] = df['Sales'] * delay_penalty_ratio * df['delayed']
    df['total_cost'] = df['shipping_cost'] + df['delay_penalty']
    
    # Cost metrics
    total_shipping_cost = df['shipping_cost'].sum()
    total_delay_penalty = df['delay_penalty'].sum()
    total_cost = df['total_cost'].sum()
    potential_savings = total_delay_penalty * 0.3  # 30% potential savings
    
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
    cost_data = {
        'Category': ['Shipping Cost', 'Delay Penalties', 'Potential Savings'],
        'Amount': [total_shipping_cost, total_delay_penalty, potential_savings]
    }
    cost_df = pd.DataFrame(cost_data)
    
    fig = px.bar(cost_df, x='Category', y='Amount', 
                title="Cost Breakdown",
                color='Category')
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.markdown("---")
    export_cost_report(df, total_shipping_cost, total_delay_penalty, total_cost, potential_savings)

def export_performance_report(df, total_shipments, delay_rate, avg_delay_days, total_sales):
    """Export performance report"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Performance to Excel"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Performance summary
                performance_data = {
                    'Metric': ['Total Shipments', 'Delay Rate', 'Avg Delay Days', 'Total Sales'],
                    'Value': [total_shipments, f"{delay_rate:.1f}%", f"{avg_delay_days:.1f}", f"${total_sales:,.0f}"]
                }
                pd.DataFrame(performance_data).to_excel(writer, sheet_name='Performance Summary', index=False)
            
            output.seek(0)
            st.download_button(
                label="📥 Download Performance Excel",
                data=output.getvalue(),
                file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📄 Generate Performance PDF"):
            report_data = {
                'summary_stats': {
                    'Total Shipments': total_shipments,
                    'Delay Rate': f"{delay_rate:.1f}%",
                    'Avg Delay Days': f"{avg_delay_days:.1f}",
                    'Total Sales': f"${total_sales:,.0f}"
                },
                'insights': [
                    f"Overall delay rate is {delay_rate:.1f}%",
                    f"Average delay duration is {avg_delay_days:.1f} days",
                    f"Total sales volume is ${total_sales:,.0f}"
                ],
                'recommendations': [
                    "Focus on reducing delay rate through process optimization",
                    "Implement proactive monitoring for high-risk shipments",
                    "Consider premium shipping for high-value orders"
                ]
            }
            
            pdf_buffer = create_pdf_report(report_data, "Performance Summary Report")
            st.download_button(
                label="📥 Download Performance PDF",
                data=pdf_buffer.getvalue(),
                file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
    
    with col3:
        st.subheader("📧 Email Performance Report")
        recipient_email = st.text_input("Email Address", key="perf_email")
        if st.button("📤 Send Performance Report"):
            if recipient_email:
                report_data = {
                    'summary_stats': {
                        'Total Shipments': total_shipments,
                        'Delay Rate': f"{delay_rate:.1f}%",
                        'Avg Delay Days': f"{avg_delay_days:.1f}",
                        'Total Sales': f"${total_sales:,.0f}"
                    },
                    'insights': [
                        f"Overall delay rate is {delay_rate:.1f}%",
                        f"Average delay duration is {avg_delay_days:.1f} days",
                        f"Total sales volume is ${total_sales:,.0f}"
                    ],
                    'recommendations': [
                        "Focus on reducing delay rate through process optimization",
                        "Implement proactive monitoring for high-risk shipments",
                        "Consider premium shipping for high-value orders"
                    ]
                }
                
                if send_email_report(report_data, recipient_email, "Performance Summary Report"):
                    st.success("✅ Performance report sent successfully!")
                else:
                    st.error("❌ Failed to send performance report.")

def export_trend_report(df, monthly_data):
    """Export trend analysis report"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Trends to Excel"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Raw Data', index=False)
                monthly_data.to_excel(writer, sheet_name='Monthly Trends', index=False)
            
            output.seek(0)
            st.download_button(
                label="📥 Download Trends Excel",
                data=output.getvalue(),
                file_name=f"trends_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📄 Generate Trends PDF"):
            report_data = {
                'summary_stats': {
                    'Total Months': len(monthly_data),
                    'Average Delay Rate': f"{monthly_data['Delay_Rate'].mean()*100:.1f}%",
                    'Total Sales': f"${monthly_data['Total_Sales'].sum():,.0f}",
                    'Trend Direction': "Increasing" if monthly_data['Delay_Rate'].iloc[-1] > monthly_data['Delay_Rate'].iloc[0] else "Decreasing"
                },
                'insights': [
                    f"Delay rate trend is {'increasing' if monthly_data['Delay_Rate'].iloc[-1] > monthly_data['Delay_Rate'].iloc[0] else 'decreasing'}",
                    f"Sales trend shows {'growth' if monthly_data['Total_Sales'].iloc[-1] > monthly_data['Total_Sales'].iloc[0] else 'decline'}",
                    f"Average monthly delay rate is {monthly_data['Delay_Rate'].mean()*100:.1f}%"
                ],
                'recommendations': [
                    "Monitor trend direction for early intervention",
                    "Analyze seasonal patterns in delay rates",
                    "Correlate sales trends with delivery performance"
                ]
            }
            
            pdf_buffer = create_pdf_report(report_data, "Trend Analysis Report")
            st.download_button(
                label="📥 Download Trends PDF",
                data=pdf_buffer.getvalue(),
                file_name=f"trends_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
    
    with col3:
        st.subheader("📧 Email Trends Report")
        recipient_email = st.text_input("Email Address", key="trend_email")
        if st.button("📤 Send Trends Report"):
            if recipient_email:
                report_data = {
                    'summary_stats': {
                        'Total Months': len(monthly_data),
                        'Average Delay Rate': f"{monthly_data['Delay_Rate'].mean()*100:.1f}%",
                        'Total Sales': f"${monthly_data['Total_Sales'].sum():,.0f}",
                        'Trend Direction': "Increasing" if monthly_data['Delay_Rate'].iloc[-1] > monthly_data['Delay_Rate'].iloc[0] else "Decreasing"
                    },
                    'insights': [
                        f"Delay rate trend is {'increasing' if monthly_data['Delay_Rate'].iloc[-1] > monthly_data['Delay_Rate'].iloc[0] else 'decreasing'}",
                        f"Sales trend shows {'growth' if monthly_data['Total_Sales'].iloc[-1] > monthly_data['Total_Sales'].iloc[0] else 'decline'}",
                        f"Average monthly delay rate is {monthly_data['Delay_Rate'].mean()*100:.1f}%"
                    ],
                    'recommendations': [
                        "Monitor trend direction for early intervention",
                        "Analyze seasonal patterns in delay rates",
                        "Correlate sales trends with delivery performance"
                    ]
                }
                
                if send_email_report(report_data, recipient_email, "Trend Analysis Report"):
                    st.success("✅ Trends report sent successfully!")
                else:
                    st.error("❌ Failed to send trends report.")

def export_risk_report(df, risk_scores):
    """Export risk assessment report"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Risk to Excel"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Risk summary
                risk_summary = {
                    'Metric': ['High Risk Shipments', 'Medium Risk Shipments', 'Low Risk Shipments', 'Average Risk Score'],
                    'Value': [
                        sum(1 for x in risk_scores if x >= 70),
                        sum(1 for x in risk_scores if 40 <= x < 70),
                        sum(1 for x in risk_scores if x < 40),
                        f"{np.mean(risk_scores):.1f}"
                    ]
                }
                pd.DataFrame(risk_summary).to_excel(writer, sheet_name='Risk Summary', index=False)
            
            output.seek(0)
            st.download_button(
                label="📥 Download Risk Excel",
                data=output.getvalue(),
                file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📄 Generate Risk PDF"):
            high_risk = sum(1 for x in risk_scores if x >= 70)
            medium_risk = sum(1 for x in risk_scores if 40 <= x < 70)
            low_risk = sum(1 for x in risk_scores if x < 40)
            
            report_data = {
                'summary_stats': {
                    'High Risk Shipments': high_risk,
                    'Medium Risk Shipments': medium_risk,
                    'Low Risk Shipments': low_risk,
                    'Average Risk Score': f"{np.mean(risk_scores):.1f}"
                },
                'insights': [
                    f"{high_risk} shipments are at high risk of delay",
                    f"{medium_risk} shipments have moderate delay risk",
                    f"Average risk score is {np.mean(risk_scores):.1f}"
                ],
                'recommendations': [
                    "Implement priority handling for high-risk shipments",
                    "Monitor medium-risk shipments closely",
                    "Develop risk mitigation strategies for vulnerable shipments"
                ]
            }
            
            pdf_buffer = create_pdf_report(report_data, "Risk Assessment Report")
            st.download_button(
                label="📥 Download Risk PDF",
                data=pdf_buffer.getvalue(),
                file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
    
    with col3:
        st.subheader("📧 Email Risk Report")
        recipient_email = st.text_input("Email Address", key="risk_email")
        if st.button("📤 Send Risk Report"):
            if recipient_email:
                high_risk = sum(1 for x in risk_scores if x >= 70)
                medium_risk = sum(1 for x in risk_scores if 40 <= x < 70)
                low_risk = sum(1 for x in risk_scores if x < 40)
                
                report_data = {
                    'summary_stats': {
                        'High Risk Shipments': high_risk,
                        'Medium Risk Shipments': medium_risk,
                        'Low Risk Shipments': low_risk,
                        'Average Risk Score': f"{np.mean(risk_scores):.1f}"
                    },
                    'insights': [
                        f"{high_risk} shipments are at high risk of delay",
                        f"{medium_risk} shipments have moderate delay risk",
                        f"Average risk score is {np.mean(risk_scores):.1f}"
                    ],
                    'recommendations': [
                        "Implement priority handling for high-risk shipments",
                        "Monitor medium-risk shipments closely",
                        "Develop risk mitigation strategies for vulnerable shipments"
                    ]
                }
                
                if send_email_report(report_data, recipient_email, "Risk Assessment Report"):
                    st.success("✅ Risk report sent successfully!")
                else:
                    st.error("❌ Failed to send risk report.")

def export_cost_report(df, total_shipping_cost, total_delay_penalty, total_cost, potential_savings):
    """Export cost analysis report"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Cost to Excel"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Cost summary
                cost_summary = {
                    'Category': ['Shipping Cost', 'Delay Penalties', 'Total Cost', 'Potential Savings'],
                    'Amount': [total_shipping_cost, total_delay_penalty, total_cost, potential_savings]
                }
                pd.DataFrame(cost_summary).to_excel(writer, sheet_name='Cost Summary', index=False)
            
            output.seek(0)
            st.download_button(
                label="📥 Download Cost Excel",
                data=output.getvalue(),
                file_name=f"cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📄 Generate Cost PDF"):
            report_data = {
                'summary_stats': {
                    'Total Shipping Cost': f"${total_shipping_cost:,.0f}",
                    'Delay Penalties': f"${total_delay_penalty:,.0f}",
                    'Total Cost': f"${total_cost:,.0f}",
                    'Potential Savings': f"${potential_savings:,.0f}"
                },
                'insights': [
                    f"Delay penalties account for {total_delay_penalty/total_cost*100:.1f}% of total costs",
                    f"Potential savings of ${potential_savings:,.0f} through optimization",
                    f"Average cost per shipment is ${total_cost/len(df):.2f}"
                ],
                'recommendations': [
                    "Focus on reducing delay penalties to improve profitability",
                    "Implement cost-effective shipping alternatives",
                    "Develop ROI-based optimization strategies"
                ]
            }
            
            pdf_buffer = create_pdf_report(report_data, "Cost Analysis Report")
            st.download_button(
                label="📥 Download Cost PDF",
                data=pdf_buffer.getvalue(),
                file_name=f"cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
    
    with col3:
        st.subheader("📧 Email Cost Report")
        recipient_email = st.text_input("Email Address", key="cost_email")
        if st.button("📤 Send Cost Report"):
            if recipient_email:
                report_data = {
                    'summary_stats': {
                        'Total Shipping Cost': f"${total_shipping_cost:,.0f}",
                        'Delay Penalties': f"${total_delay_penalty:,.0f}",
                        'Total Cost': f"${total_cost:,.0f}",
                        'Potential Savings': f"${potential_savings:,.0f}"
                    },
                    'insights': [
                        f"Delay penalties account for {total_delay_penalty/total_cost*100:.1f}% of total costs",
                        f"Potential savings of ${potential_savings:,.0f} through optimization",
                        f"Average cost per shipment is ${total_cost/len(df):.2f}"
                    ],
                    'recommendations': [
                        "Focus on reducing delay penalties to improve profitability",
                        "Implement cost-effective shipping alternatives",
                        "Develop ROI-based optimization strategies"
                    ]
                }
                
                if send_email_report(report_data, recipient_email, "Cost Analysis Report"):
                    st.success("✅ Cost report sent successfully!")
                else:
                    st.error("❌ Failed to send cost report.")

if __name__ == "__main__":
    main() 