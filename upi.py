import streamlit as st
import pandas as pd
import re
import pdfplumber
import plotly.express as px
from datetime import datetime
import google.generativeai as genai
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go

# Streamlit App Setup
st.set_page_config(page_title="UPI Statement Analyzer", layout="wide")
st.title("📊 UPI Transaction Analytics with AI Insights")

# Configure Gemini API
GEMINI_API_KEY = st.secrets["gemini_api_key"]  # Store your API key in Streamlit secrets
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# File Upload
uploaded_file = st.file_uploader("Upload your UPI Statement PDF", type="pdf")

if uploaded_file:
    @st.cache_data
    def extract_transactions(pdf_file):
        all_lines = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_lines.extend([line.strip() for line in text.split('\n') if line.strip()])

        transactions = []
        current_transaction = {}

        for line in all_lines:
            try:
                # Check for date line (start of new transaction)
                date_match = re.match(r'^([A-Za-z]{3} \d{2}, \d{4})', line)
                if date_match:
                    if current_transaction:
                        transactions.append(current_transaction)
                        current_transaction = {}

                    parts = line.split(maxsplit=3)
                    current_transaction['Date'] = parts[0] + ' ' + parts[1].rstrip(',')
                    current_transaction['Time'] = parts[2] if len(parts) > 2 else ''

                    # Extract transaction type
                    if 'Received from' in line:
                        current_transaction['Type'] = 'Credit'
                        desc_start = line.find('Received from') + len('Received from')
                        current_transaction['Description'] = line[desc_start:].split('Credit')[0].strip()
                    elif 'Paid to' in line:
                        current_transaction['Type'] = 'Debit'
                        desc_start = line.find('Paid to') + len('Paid to')
                        current_transaction['Description'] = line[desc_start:].split('Debit')[0].strip()
                    elif 'Bill paid' in line:
                        current_transaction['Type'] = 'Debit'
                        desc_start = line.find('Bill paid') + len('Bill paid')
                        current_transaction['Description'] = line[desc_start:].split('Debit')[0].strip().lstrip('-').strip()

                # Check for amount line
                amount_match = re.search(r'(\d+,\d+\.\d{2}|\d+\.\d{2})$', line.strip())
                if amount_match and 'Amount' not in current_transaction:
                    amount_str = amount_match.group(1).replace(',', '')
                    current_transaction['Amount'] = float(amount_str)

                # Check for UTR line
                if 'UTR No :' in line:
                    current_transaction['UTR'] = line.split('UTR No :')[-1].strip()

            except Exception as e:
                continue

        if current_transaction:
            transactions.append(current_transaction)

        return pd.DataFrame(transactions)

    @st.cache_data
    def enhance_data(df):
        # Combine Date and Time
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].astype(str), format='%b %d %Y')
        
        # Handle missing values
        df['Type'] = df['Type'].fillna('Unknown')
        df['Amount'] = df['Amount'].fillna(0)
        df['Description'] = df['Description'].fillna('No Description')
        df['UTR'] = df['UTR'].fillna('No UTR')
        
        # Extract time components
        df['Month'] = df['DateTime'].dt.month_name()
        df['Year'] = df['DateTime'].dt.year
        df['Day_of_Month'] = df['DateTime'].dt.day
        df['Day_of_Week'] = df['DateTime'].dt.day_name()
        df['Hour'] = df['DateTime'].dt.hour
        
        # Categorize transactions
        categories = {
            'Grocery': ['bigbasket', 'grocery', 'supermarket', 'mart'],
            'Food': ['zomato', 'swiggy', 'restaurant', 'cafe'],
            'Shopping': ['amazon', 'flipkart', 'myntra'],
            'Bills': ['electricity', 'bill', 'mobile', 'internet'],
            'Entertainment': ['netflix', 'prime', 'movie']
        }
        
        def categorize(desc):
            desc = str(desc).lower()
            for cat, keywords in categories.items():
                if any(keyword in desc for keyword in keywords):
                    return cat
            return 'Other'
        
        df['Category'] = df['Description'].apply(categorize)
        
        return df

    def generate_ai_insights(df):
        """Generate AI-powered financial insights using Gemini"""
        try:
            # Prepare data summary for AI
            total_credit = df[df['Type']=='Credit']['Amount'].sum()
            total_debit = df[df['Type']=='Debit']['Amount'].sum()
            top_categories = df[df['Type']=='Debit'].groupby('Category')['Amount'].sum().nlargest(3).to_dict()
            frequent_merchants = df['Description'].value_counts().head(3).to_dict()
        
            prompt = f"""
            Act as a financial advisor analyzing UPI transaction data with these key metrics:
            - Total Credits: ₹{total_credit:,.2f}
            - Total Debits: ₹{total_debit:,.2f}
            - Top Spending Categories: {top_categories}
            - Frequent Merchants: {frequent_merchants}
        
            Provide specific, actionable insights in this format:
        
            💡 Savings Opportunities:
            - Identify 2-3 specific areas where the user could reduce spending
            - Suggest practical alternatives for each
        
            🔮 Future Planning:
            - Predict upcoming recurring payments based on patterns
            - Estimate monthly financial commitments
        
            🚀 Smart Money Moves:
            - Recommend 2-3 specific financial products/services that could help
            - Suggest budgeting strategies tailored to this spending pattern
        
            Keep responses concise, practical, and personalized. Use bullet points for readability.
            """
        
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ Could not generate insights: {str(e)}"

    # Process PDF
    raw_df = extract_transactions(uploaded_file)
    cleaned_df = enhance_data(raw_df)
    
    # Display Data
    st.subheader("🧾 Raw Transaction Data")
    st.dataframe(raw_df, use_container_width=True)
    
    st.subheader("✨ Enhanced Transaction Data")
    st.dataframe(cleaned_df, use_container_width=True)
    
    # Analytics Section
    st.header("📈 Transaction Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Transactions", len(cleaned_df))
        fig = px.pie(cleaned_df, names='Type', title='Transaction Types')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Total Amount", f"₹{cleaned_df['Amount'].sum():,.2f}")
        fig = px.histogram(cleaned_df, x='Amount', nbins=20, title='Amount Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Time Analysis
    st.subheader("⏰ Transaction Timing")
    tab1, tab2 = st.tabs(["Monthly Trends", "Daily Patterns"])
    
    with tab1:
        monthly = cleaned_df.groupby(['Year', 'Month', 'Type'])['Amount'].sum().reset_index()
        fig = px.bar(monthly, x='Month', y='Amount', color='Type', 
                     barmode='group', title='Monthly Transactions')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        daily = cleaned_df.groupby(['Day_of_Week', 'Type'])['Amount'].sum().reset_index()
        fig = px.line(daily, x='Day_of_Week', y='Amount', color='Type',
                      title='Weekly Transaction Patterns')
        st.plotly_chart(fig, use_container_width=True)
    
    cleaned_df['Month_Period'] = pd.cut(cleaned_df['Day_of_Month'], 
                                    bins=[0, 10, 20, 31], 
                                    labels=['Early', 'Mid', 'Late'])
    # Category Analysis
    st.subheader("🛍️ Spending Categories")
    debit_df = cleaned_df[cleaned_df['Type'] == 'Debit']
    fig = px.sunburst(debit_df, path=['Category', 'Description'], values='Amount',
                      title='Debit Transaction Breakdown')
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced Analytics Section
    st.header("💰 Advanced Financial Analytics")
    
    # 1. Category Spending Deep Dive
    st.subheader("🔍 Category Spending Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        # Top spending categories
        category_spend = debit_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        fig = px.bar(category_spend, 
                    title='Total Spending by Category (₹)',
                    labels={'value': 'Amount (₹)', 'index': 'Category'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average transaction by category
        category_avg = debit_df.groupby('Category')['Amount'].mean().sort_values(ascending=False)
        fig = px.bar(category_avg, 
                    title='Average Transaction by Category (₹)',
                    labels={'value': 'Average Amount (₹)', 'index': 'Category'})
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Monthly Spending Patterns
    st.subheader("📅 Monthly Spending Patterns")
    
    # Create month periods
    cleaned_df['Month_Period'] = pd.cut(cleaned_df['Day_of_Month'], 
                               bins=[1, 10, 20, 31],
                               labels=['Start (1-10)', 'Middle (11-20)', 'End (21-31)'])
    
    period_tab1, period_tab2, period_tab3 = st.tabs(["By Amount", "By Frequency", "Trend Analysis"])
    
    with period_tab1:
        period_amount = debit_df.groupby('Month_Period')['Amount'].sum()
        fig = px.pie(period_amount, 
                    names=period_amount.index,
                    values='Amount',
                    title='Spending Distribution Across Month')
        st.plotly_chart(fig, use_container_width=True)
    
    with period_tab2:
        period_count = debit_df.groupby('Month_Period').size()
        fig = px.bar(period_count,
                    title='Transaction Frequency Across Month',
                    labels={'value': 'Number of Transactions', 'index': 'Period'})
        st.plotly_chart(fig, use_container_width=True)
    
    with period_tab3:
        # Time series decomposition
        try:
            daily_spend = debit_df.set_index('DateTime').resample('D')['Amount'].sum().fillna(0)
            result = seasonal_decompose(daily_spend, model='additive', period=30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, name='Trend'))
            fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, name='Seasonal'))
            fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, name='Residual'))
            fig.update_layout(title='Spending Trend Decomposition')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Couldn't perform trend analysis: {str(e)}")
    
    # 3. Financial Health Metrics
    st.subheader("💵 Financial Health Indicators")
    
    # Calculate metrics
    credit_total = cleaned_df[cleaned_df['Type']=='Credit']['Amount'].sum()
    debit_total = cleaned_df[cleaned_df['Type']=='Debit']['Amount'].sum()
    savings_ratio = (credit_total - debit_total) / credit_total if credit_total > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Income (Credits)", f"₹{credit_total:,.2f}")
    col2.metric("Expenses (Debits)", f"₹{debit_total:,.2f}")
    col3.metric("Savings Ratio", f"{savings_ratio:.1%}")
    
    # 4. Recurring Expenses Detection
    st.subheader("🔄 Recurring Expense Analysis")
    
    # Find potential recurring payments
    recurring_candidates = debit_df.groupby(['Description', 'Category'])['Amount'].agg(['mean', 'count'])
    recurring_candidates = recurring_candidates[recurring_candidates['count'] > 1].sort_values('count', ascending=False)
    
    if len(recurring_candidates) > 0:
        st.write("Potential recurring expenses:")
        st.dataframe(recurring_candidates.style.format({'mean': "₹{:.2f}"}))
        
        # Visualize recurring expenses
        top_recurring = recurring_candidates.head(10).reset_index()
        fig = px.bar(top_recurring, 
                     x='Description', 
                     y='mean', 
                     color='Category',
                     title='Top Recurring Expenses (Avg ₹)', 
                     labels={'mean': 'Average Amount (₹)'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recurring expenses detected.")

    # 5. Spending Alerts
    st.subheader("⚠️ Spending Alerts")
    
    # Detect unusual spending
    avg_spend = debit_df['Amount'].mean()
    std_spend = debit_df['Amount'].std()
    unusual_spending = debit_df[debit_df['Amount'] > (avg_spend + 2*std_spend)]
    
    if len(unusual_spending) > 0:
        st.warning(f"Found {len(unusual_spending)} unusually large transactions:")
        st.dataframe(unusual_spending.sort_values('Amount', ascending=False))
    else:
        st.success("No unusually large transactions detected")
    
    # AI Insights Section
    st.header("🤖 AI-Powered Financial Insights")
    
    with st.spinner("Generating personalized insights..."):
        insights = generate_ai_insights(cleaned_df)
    
    st.markdown(insights)
    
    st.divider()
    
    # Interactive Q&A
    st.subheader("❓ Ask a Question About Your Spending")
    user_question = st.text_input("Type your question here and press Enter:")
    
    if user_question:
        with st.spinner("Analyzing your question..."):
            q_prompt = f"""
            Based on this transaction data:
            {cleaned_df.head().to_string()}
            
            And these overall stats:
            - Total Credits: ₹{cleaned_df[cleaned_df['Type']=='Credit']['Amount'].sum():,.2f}
            - Total Debits: ₹{cleaned_df[cleaned_df['Type']=='Debit']['Amount'].sum():,.2f}
            
            Answer this user question concisely and helpfully: {user_question}
            """
            
            q_response = model.generate_content(q_prompt)
            st.markdown(f"**AI Answer:** {q_response.text}")

# How to run:
# streamlit run upi_analyzer.py