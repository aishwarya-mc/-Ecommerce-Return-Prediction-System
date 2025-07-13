import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def main():
    
    model = joblib.load("return_reason_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_reason = joblib.load("label_encoder_reason.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    
   
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
        
      
        .reason-card {
           linear-gradient(135deg, #0f0c29, #302b63, #24243e);
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
        
        .reason-card::before {
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
        
        .reason-card h2 {
            font-weight: 600;
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .reason-card h3 {
            font-weight: 500;
            font-size: 1.4rem;
            margin-bottom: 1rem;
            opacity: 0.9;
        }
        
        .reason-card p {
            font-size: 1.1rem;
            opacity: 0.8;
            line-height: 1.6;
        }
        
        
        .reason-item {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            margin: 0.8rem 0;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .reason-item:hover {
            transform: translateX(5px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
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

    
    if 'return_reason_prediction_made' not in st.session_state:
        st.session_state.return_reason_prediction_made = False
    
    if 'return_reason_results' not in st.session_state:
        st.session_state.return_reason_results = None

    st.title(" Predict Product Return Reason")

   
    st.markdown("###  Return Reason Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Return Reasons", "5 Categories", "📋")
    with col2:
        st.metric("Model Accuracy", "High", "🎯")
    with col3:
        st.metric("Top 3 Predictions", "Ranked", "🏆")

   
    st.markdown("### Enter Order Details")
    
   
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("####  Product Information")
        product_price = st.number_input("Product Price ($)", min_value=0.01, value=100.0, step=1.0, help="Enter the product price", key="reason_price_input")
        discount_applied = st.number_input("Discount Applied ($)", min_value=0.0, value=10.0, step=1.0, help="Enter any discount applied", key="reason_discount_input")
        order_quantity = st.number_input("Order Quantity", min_value=1, value=1, step=1, help="Number of items ordered", key="reason_quantity_input")
        days_to_return = st.number_input("Days to Return", value=-1, step=1, help="Days until return (-1 if not returned)", key="reason_days_input")

    with col2:
        st.markdown("####  Product & Customer Details")
        product_category = st.selectbox("Product Category", ['Electronics', 'Apparel', 'Books', 'Furniture'], help="Select the product category", key="reason_category_select")
        user_gender = st.selectbox("Customer Gender", ['Male', 'Female'], help="Select customer gender", key="reason_gender_select")
        payment_method = st.selectbox("Payment Method", ['Credit Card', 'COD', 'UPI'], help="Select payment method", key="reason_payment_select")
        shipping_method = st.selectbox("Shipping Method", ['Standard', 'Express'], help="Select shipping method", key="reason_shipping_select")

    
    if st.button(" Predict Return Reason", use_container_width=True, key="reason_predict_button"):
        st.session_state.return_reason_prediction_made = True
        

        user_input = {
            'Product_Price': product_price,
            'Discount_Applied': discount_applied,
            'Order_Quantity': order_quantity,
            'Days_to_Return': days_to_return,
            'Total_Amount': product_price * order_quantity,
            f'Product_Category_{product_category}': 1,
            f'User_Gender_{user_gender}': 1,
            f'Payment_Method_{payment_method}': 1,
            f'Shipping_Method_{shipping_method}': 1
        }

     
        input_df = pd.DataFrame([user_input])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        
        try:
          
            numeric_cols = ['Product_Price', 'Discount_Applied', 'Total_Amount']
            numeric_data = input_df[numeric_cols].values
            scaled_numeric = scaler.transform(numeric_data)
            input_df[numeric_cols] = scaled_numeric

         
            probs = model.predict_proba(input_df)[0]
            top3_indices = probs.argsort()[::-1][:3]
            top3_reasons = le_reason.inverse_transform(top3_indices)
            top3_probs = probs[top3_indices]

            st.session_state.return_reason_results = {
                'top3_reasons': top3_reasons,
                'top3_probs': top3_probs,
                'probs': probs,
                'top_reason': top3_reasons[0],
                'top_prob': top3_probs[0]
            }
            st.rerun()

        except Exception as e:
            st.error("⚠️ Error during prediction. Please check input values.")
            st.exception(e)

  
    if st.session_state.return_reason_prediction_made and st.session_state.return_reason_results:
        results = st.session_state.return_reason_results
        top3_reasons = results['top3_reasons']
        top3_probs = results['top3_probs']
        top_reason = results['top_reason']
        top_prob = results['top_prob']
        probs = results['probs']
        
    
        st.markdown("###  Return Reason Analysis")
        
       
        st.markdown(f"""
        <div class="reason-card">
            <h2> Most Likely Return Reason</h2>
            <h3>{top_reason}</h3>
            <p>Confidence: {top_prob:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

     
        st.markdown("### Top 3 Return Reasons")
        
   
        fig = go.Figure(data=[
            go.Bar(
                x=top3_reasons,
                y=top3_probs,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                text=[f'{prob:.1%}' for prob in top3_probs],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Return Reason Probability Distribution",
            xaxis_title="Return Reason",
            yaxis_title="Probability",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True, key="reason_chart")

        # Detailed breakdown
        st.markdown("###  Detailed Analysis")
        
        for i, (reason, prob) in enumerate(zip(top3_reasons, top3_probs)):
            rank_emoji = ["1:", "2:", "3:"][i]
            st.markdown(f"""
            <div class="reason-item">
                <h4>{rank_emoji} {reason}</h4>
                <p><strong>Probability:</strong> {prob:.1%}</p>
                <p><strong>Rank:</strong> #{i+1}</p>
            </div>
            """, unsafe_allow_html=True)

       
        st.markdown("### Model Confidence")
        confidence_score = max(probs)
        st.metric("Overall Confidence", f"{confidence_score:.1%}")

       
        st.markdown("###  Insights")
        if top_reason == "Defective":
            st.info(" This suggests a potential product quality issue that may need attention.")
        elif top_reason == "Size too small":
            st.info(" Consider improving size guides or offering more size options.")
        elif top_reason == "Changed mind":
            st.info(" This is a common customer behavior, consider improving product descriptions.")
        elif top_reason == "Not as described":
            st.info(" Product descriptions may need improvement to match customer expectations.")
        elif top_reason == "Wrong item":
            st.info(" There may be issues with order fulfillment or product labeling.")

if __name__ == "__main__":
    main()
