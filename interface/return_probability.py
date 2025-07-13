import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def main():
   
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
        
       
        .prediction-card {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
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
        
        .prediction-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            animation: shimmer 3s infinite;
        }
        
        @keyframes cardFloat {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .prediction-card h2 {
            font-weight: 600;
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .prediction-card h3 {
            font-weight: 500;
            font-size: 1.4rem;
            margin-bottom: 1rem;
            opacity: 0.9;
        }
        
        .prediction-card p {
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
        
       
        .feature-importance {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            padding: 1.2rem;
            border-radius: 12px;
            border-left: 4px solid #667eea;
            border-right: 4px solid #764ba2;
            margin: 0.8rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .feature-importance::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        .feature-importance:hover {
            transform: translateX(5px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        .feature-importance strong {
            color: #e8e8e8;
            font-weight: 600;
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
        
       
        .stColumns {
            gap: 2rem;
        }
      
        .loading-text {
            color: #667eea;
            font-weight: 500;
            font-size: 1.1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    
    if 'return_prob_prediction_made' not in st.session_state:
        st.session_state.return_prob_prediction_made = False
    
    if 'return_prob_results' not in st.session_state:
        st.session_state.return_prob_results = None

   
    @st.cache_data
    def load_and_train_model():
        with st.spinner("Loading and training model..."):
            progress_bar = st.progress(0)
            
            df = pd.read_csv("ecommerce_returns_synthetic_data_realistic.csv")
            progress_bar.progress(25)
            
          
            order_time_features = [
                'Product_Category', 'Product_Price', 'Order_Quantity', 'User_Age', 'User_Gender',
                'User_Location', 'Payment_Method', 'Shipping_Method', 'Discount_Applied'
            ]
            df = df.dropna(subset=['Return_Status'])
            df = df[order_time_features + ['Return_Status']]
            progress_bar.progress(50)
            
            for col in ['Product_Price', 'Order_Quantity', 'User_Age', 'Discount_Applied']:
                df[col] = df[col].fillna(df[col].median())
            categorical_cols = ['Product_Category', 'User_Gender', 'User_Location', 'Payment_Method', 'Shipping_Method']
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            progress_bar.progress(75)
            
            scaler = StandardScaler()
            numeric_cols = ['Product_Price', 'Order_Quantity', 'User_Age', 'Discount_Applied']
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            df['Return_Status'] = df['Return_Status'].map({'Returned': 1, 'Not Returned': 0})
            X = df.drop('Return_Status', axis=1)
            y = df['Return_Status']
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            progress_bar.progress(100)
            
            importances = pd.Series(model.feature_importances_, index=X.columns)
            important_cols = importances[importances > 0.01].sort_values(ascending=False).index.tolist()
            model.fit(X_train[important_cols], y_train)
            return model, scaler, important_cols, X_train.columns.tolist()

    
    model, scaler, important_cols, full_feature_list = load_and_train_model()
    
    if not st.session_state.model_loaded:
        st.success(" Model loaded successfully!")
        st.session_state.model_loaded = True

    
    st.markdown("###  Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Orders", "10,000+", "📦")
    with col2:
        st.metric("Return Rate", "15.3%", "📈")
    with col3:
        st.metric("Features Used", f"{len(important_cols)}", "🔧")

   
    st.markdown("###  Enter Order Details")
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("####  Product Information")
        user_input = {
            'Product_Price': st.number_input("Product Price ($)", value=100.0, min_value=0.01, help="Enter the product price in dollars", key="price_input"),
            'Discount_Applied': st.number_input("Discount Applied ($)", value=10.0, min_value=0.0, help="Enter any discount applied to the order", key="discount_input"),
            'Order_Quantity': st.number_input("Order Quantity", value=1, min_value=1, help="Number of items ordered", key="quantity_input"),
            'User_Age': st.number_input("Customer Age", value=30, min_value=10, max_value=100, help="Customer's age", key="age_input")
        }

    with col2:
        st.markdown("####  Customer Information")
        
        cat_fields = {
            'Product_Category': ['Electronics', 'Apparel', 'Books', 'Furniture', 'Clothing'],
            'User_Gender': ['Male', 'Female'],
            'User_Location': ['City1', 'City2', 'City3', 'City4', 'City5'],
            'Payment_Method': ['Credit Card', 'COD', 'UPI'],
            'Shipping_Method': ['Standard', 'Express'],
        }

        for field, options in cat_fields.items():
            selected = st.selectbox(field.replace('_', ' '), options, help=f"Select the {field.lower()}", key=f"{field}_select")
            col_name = f"{field}_{selected}"
            user_input[col_name] = 1

   
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)

    
    numeric_cols = ['Product_Price', 'Order_Quantity', 'User_Age', 'Discount_Applied']  # Same order as training
    available_numeric_cols = [col for col in numeric_cols if col in input_df.columns]

    
    if set(numeric_cols).issubset(input_df.columns):
        
        numeric_data = input_df[numeric_cols].values
        scaled_numeric = scaler.transform(numeric_data)
        input_df[numeric_cols] = scaled_numeric
    else:
        st.error(f"Missing one or more numeric columns required for scaling: {numeric_cols}")
        st.stop()

  
    input_df = input_df.reindex(columns=full_feature_list, fill_value=0)

    
    input_df = input_df[important_cols]

    
    probs = model.predict_proba(input_df)[0]
    pred = np.argmax(probs)
    labels = ['Not Returned', 'Returned']

    
    if st.button(" Predict Return Status", use_container_width=True, key="predict_button"):
        st.session_state.return_prob_prediction_made = True
        st.session_state.return_prob_results = {
            'probs': probs,
            'pred': pred,
            'labels': labels,
            'important_cols': important_cols,
            'model': model
        }
        st.rerun()

 
    if st.session_state.return_prob_prediction_made and st.session_state.return_prob_results:
        results = st.session_state.return_prob_results
        probs = results['probs']
        pred = results['pred']
        labels = results['labels']
        important_cols = results['important_cols']
        model = results['model']
        
        
        st.markdown("### Prediction Results")
        
        
        if pred == 0:
            st.markdown(f"""
            <div class="prediction-card">
                <h2>Order Likely to be Kept</h2>
                <h3>Confidence: {probs[pred]:.1%}</h3>
                <p>This order shows characteristics of orders that are typically kept by customers.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card">
                <h2>⚠️ Order Likely to be Returned</h2>
                <h3>Confidence: {probs[pred]:.1%}</h3>
                <p>This order shows characteristics of orders that are typically returned by customers.</p>
            </div>
            """, unsafe_allow_html=True)

       
        st.markdown("###  Probability Breakdown")
        
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Not Returned', 'Returned'],
                y=[probs[0], probs[1]],
                marker_color=['#2E8B57', '#DC143C'],
                text=[f'{probs[0]:.1%}', f'{probs[1]:.1%}'],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Return Probability Distribution",
            xaxis_title="Prediction",
            yaxis_title="Probability",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True, key="prob_chart")

        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("####  Detailed Probabilities")
            for i, label in enumerate(labels):
                st.metric(f"{label}", f"{probs[i]:.1%}")

        with col2:
            st.markdown("####  Model Confidence")
            confidence_score = max(probs)
            st.metric("Model Confidence", f"{confidence_score:.1%}")

        
        st.markdown("###  Top Influential Features")
        
        importances = pd.Series(model.feature_importances_, index=important_cols)
        top_features = importances.sort_values(ascending=False).head(5)
        
        
        fig_importance = px.bar(
            x=top_features.values,
            y=top_features.index,
            orientation='h',
            title="Top 5 Most Influential Features",
            labels={'x': 'Importance Score', 'y': 'Feature'}
        )
        
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True, key="importance_chart")

       
        st.markdown("####  Feature Importance Details")
        for i, (name, score) in enumerate(top_features.items()):
            st.markdown(f"""
            <div class="feature-importance">
                <strong>{name}</strong> — Importance: {score:.2%}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
