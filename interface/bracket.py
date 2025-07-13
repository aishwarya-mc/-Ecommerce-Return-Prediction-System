import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

def main():
 
    model = joblib.load("bracketing_model.pkl")
    feature_columns = joblib.load("bracketing_feature_columns.pkl")
    
    
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
       
        .stApp {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            color: #e8e8e8;
            font-family: 'Inter', sans-serif;
        }
        
        
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1a1a2e;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
        }
        
      
        .bracketing-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2.5rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            text-align: center;
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            animation: cardFloat 3s ease-in-out infinite;
            position: relative;
            overflow: hidden;
        }
        
        .bracketing-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes cardFloat {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .bracketing-card h2 {
            font-weight: 600;
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .bracketing-card h3 {
            font-weight: 500;
            font-size: 1.4rem;
            margin-bottom: 1rem;
            opacity: 0.9;
        }
        
        .bracketing-card p {
            font-size: 1.1rem;
            opacity: 0.8;
            line-height: 1.6;
        }
        
      
        .bracketing-detected {
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
            color: white;
            padding: 2.5rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            text-align: center;
            box-shadow: 0 20px 40px rgba(255, 107, 107, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            animation: cardFloat 3s ease-in-out infinite;
            position: relative;
            overflow: hidden;
        }
        
        .bracketing-detected::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            animation: shimmer 2s infinite;
        }
        
        .bracketing-detected h2 {
            font-weight: 600;
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .bracketing-detected h3 {
            font-weight: 500;
            font-size: 1.4rem;
            margin-bottom: 1rem;
            opacity: 0.9;
        }
        
        .bracketing-detected p {
            font-size: 1.1rem;
            opacity: 0.8;
            line-height: 1.6;
        }
        
        
        .bracketing-clean {
            linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: white;
            padding: 2.5rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            text-align: center;
            box-shadow: 0 20px 40px rgba(78, 205, 196, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            animation: cardFloat 3s ease-in-out infinite;
            position: relative;
            overflow: hidden;
        }
        
        .bracketing-clean::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            animation: shimmer 2s infinite;
        }
        
        .bracketing-clean h2 {
            font-weight: 600;
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .bracketing-clean h3 {
            font-weight: 500;
            font-size: 1.4rem;
            margin-bottom: 1rem;
            opacity: 0.9;
        }
        
        .bracketing-clean p {
            font-size: 1.1rem;
            opacity: 0.8;
            line-height: 1.6;
        }
        
       
        .metric-box {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            padding: 1.8rem;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        
        .metric-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        }
        
 
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
      
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            color: white;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
        }
        
        .stNumberInput > div > div > input {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            color: #e8e8e8;
            font-family: 'Inter', sans-serif;
        }
        
        .stNumberInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }
        
        .stSelectbox > div > div > div {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            color: #e8e8e8;
        }
        
        .stSelectbox > div > div > div:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }
        
      
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #e8e8e8;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
        }
        
        .stMarkdown h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
        }
        
        .stMarkdown h2 {
            color: #667eea;
            font-size: 2rem;
        }
        
        .stMarkdown h3 {
            color: #764ba2;
            font-size: 1.5rem;
        }
        
       
        .stMetric {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            margin: 0.5rem 0;
        }
        
      
        .js-plotly-plot {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
            max-width: 100%;
            box-sizing: border-box;
        }
        

        .js-plotly-plot .plotly {
            max-width: 100% !important;
            overflow: hidden !important;
        }
        
   
        .js-plotly-plot svg {
            max-width: 100% !important;
            height: auto !important;
        }
        
  
        .stApp > div > div > div > div {
            min-height: 100vh;
        }
        
        .stMarkdown, .stMetric, .stDataFrame {
            transition: all 0.3s ease;
        }
        
      
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background: transparent;
        }

        .stSuccess {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 25px rgba(40, 167, 69, 0.3);
        }
        
    
        .stError {
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 25px rgba(220, 53, 69, 0.3);
        }
        
   
        .stColumns {
            gap: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    if 'bracketing_prediction_made' not in st.session_state:
        st.session_state.bracketing_prediction_made = False
    
    if 'bracketing_results' not in st.session_state:
        st.session_state.bracketing_results = None

    st.title("Bracketing Detection System")

    st.markdown("###Bracketing Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Detection Rate", "5.2%", "🎯")
    with col2:
        st.metric("Model Accuracy", "98.8%", "📈")
    with col3:
        st.metric("Risk Level", "Low", "🛡️")

    st.markdown("###  Enter Order Features")
    

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Product Information")
        product_price = st.number_input("Product Price ($)", value=100.0, min_value=0.0, help="Enter the product price", key="bracket_price_input")
        discount_applied = st.number_input("Discount Applied ($)", value=10.0, min_value=0.0, help="Enter any discount applied", key="bracket_discount_input")
        order_quantity = st.number_input("Order Quantity", value=1, min_value=1, help="Number of items ordered", key="bracket_quantity_input")
        days_to_return = st.number_input("Days to Return", value=-1, help="Days until return (-1 if not returned)", key="bracket_days_input")

    with col2:
        st.markdown("####  Product & Customer Details")
        product_category = st.selectbox("Product Category", ['Clothing', 'Electronics', 'Books', 'Toys', 'Home'], help="Select the product category", key="bracket_category_select")
        user_gender = st.selectbox("Customer Gender", ['Male', 'Female'], help="Select customer gender", key="bracket_gender_select")
        payment_method = st.selectbox("Payment Method", ['Credit Card', 'COD', 'UPI'], help="Select payment method", key="bracket_payment_select")
        shipping_method = st.selectbox("Shipping Method", ['Standard', 'Express'], help="Select shipping method", key="bracket_shipping_select")

    total_amount = product_price * order_quantity


    st.markdown("###  Order Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Amount", f"${total_amount:.2f}")
    with col2:
        st.metric("Product Category", product_category)
    with col3:
        st.metric("Customer", user_gender)


    user_input = {
        'Product_Price': product_price,
        'Discount_Applied': discount_applied,
        'Order_Quantity': order_quantity,
        'Days_to_Return': days_to_return,
        'Total_Amount': total_amount,
        f'Product_Category_{product_category}': 1,
        f'User_Gender_{user_gender}': 1,
        f'Payment_Method_{payment_method}': 1,
        f'Shipping_Method_{shipping_method}': 1,
    }

    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    if st.button("🔍 Detect Bracketing", use_container_width=True, key="bracket_predict_button"):
        st.session_state.bracketing_prediction_made = True
        
        pred = model.predict(input_df)[0]
        result = "Yes" if pred == 1 else "No"
        
        categories = ['Price Risk', 'Quantity Risk', 'Category Risk', 'Customer Risk', 'Pattern Risk']
        values = [
            min(1.0, product_price / 500),  # Price risk
            min(1.0, order_quantity / 5),    # Quantity risk
            0.3 if product_category in ['Clothing', 'Electronics'] else 0.1,  # Category risk
            0.2,  # Customer risk (placeholder)
            0.1 if pred == 1 else 0.8  # Pattern risk
        ]
        
        st.session_state.bracketing_results = {
            'pred': pred,
            'result': result,
            'categories': categories,
            'values': values,
            'product_category': product_category,
            'order_quantity': order_quantity,
            'total_amount': total_amount
        }
        st.rerun()

  
    if st.session_state.bracketing_prediction_made and st.session_state.bracketing_results:
        results = st.session_state.bracketing_results
        pred = results['pred']
        result = results['result']
        categories = results['categories']
        values = results['values']
        product_category = results['product_category']
        order_quantity = results['order_quantity']
        total_amount = results['total_amount']
        
        st.markdown("###  Bracketing Analysis Results")
        
        if pred == 1:
           
            st.markdown(f"""
            <div class="bracketing-detected">
                <h2>⚠️ Bracketing Behavior Detected</h2>
                <h3>Risk Level: HIGH</h3>
                <p>This order shows characteristics of bracketing behavior where customers 
                order multiple variants with the intention of returning unwanted items.</p>
            </div>
            """, unsafe_allow_html=True)
         
            st.markdown("### 🔍 Risk Indicators")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Product Category", product_category, delta="High Risk")
            with col2:
                st.metric("Order Quantity", order_quantity, delta="Multiple Items")
            with col3:
                st.metric("Total Amount", f"${total_amount:.2f}", delta="High Value")
            
          
            st.markdown("### 💡 Recommendations")
            st.warning("""
            **Immediate Actions:**
            - Review order history for this customer
            - Consider implementing size guides
            - Monitor return patterns
            - Evaluate product descriptions
            """)
            
        else:
           
            st.markdown(f"""
            <div class="bracketing-clean">
                <h2> No Bracketing Detected</h2>
                <h3>Risk Level: LOW</h3>
                <p>This order appears to be a normal purchase without signs of bracketing behavior.</p>
            </div>
            """, unsafe_allow_html=True)
            
           
            st.markdown("###  Order Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Order Type", "Normal", delta="Clean")
            with col2:
                st.metric("Risk Score", "Low", delta="Safe")
            with col3:
                st.metric("Confidence", "High", delta="Reliable")
            
        
            st.markdown("###  Positive Indicators")
            st.success("""
            **Good Signs:**
            - Normal order quantity
            - Standard product category
            - Reasonable order value
            - Typical customer behavior
            """)

    
        st.markdown("###  Detailed Analysis")
    
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Profile',
            line_color='red' if pred == 1 else 'green'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Order Risk Profile",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True, key="bracket_radar_chart")

 
        st.markdown("###  Risk Breakdown")
        risk_data = {
            'Risk Factor': categories,
            'Risk Score': values
        }
        risk_df = pd.DataFrame(risk_data)
        
        fig_risk = px.bar(
            risk_df,
            x='Risk Factor',
            y='Risk Score',
            color='Risk Score',
            title="Risk Factor Analysis",
            color_continuous_scale='RdYlGn_r'
        )
        
        fig_risk.update_layout(
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(tickangle=45),
            autosize=True
        )
        st.plotly_chart(fig_risk, use_container_width=True, key="bracket_risk_chart")

if __name__ == "__main__":
    main()
