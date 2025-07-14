import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="Walmart Return Prediction System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        border: solid 3px #302b63;
    }
    
    .metric-card {
        background:linear-gradient(135deg, #1e1e2f, #302b63, #7209b7, #560bad);
        padding: 1.8rem;
        border-radius: 10px;
        border: 1px solid #302b63;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #+, #ff7f0e);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .feature-importance {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
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
    }
</style>
""", unsafe_allow_html=True)


if 'current_page' not in st.session_state:
    st.session_state.current_page = "Return Probability"

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Header
st.markdown("""
<div class="main-header">
    <h1>🛒 E-commerce Return Prediction System</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">Advanced ML-powered analytics for return prediction, reason analysis, and bracketing detection</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("###  Navigation")
    selected = option_menu(
        menu_title=None,
        options=["Return Probability", "Return Reason", "Bracketing Detection"],
        menu_icon="cast",
        default_index=0,
        key="navigation_menu",
        styles={
            "container": {"padding": "0!important", "background-color": "#531E90"},
            "icon": {"color": "orange", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#999"},
            "nav-link-selected": {"background-color": "#392952"},
        }
    )
    
   
    if selected != st.session_state.current_page:
        st.session_state.current_page = selected
        st.session_state.model_loaded = False 


main_container = st.container()

with main_container:
    if st.session_state.current_page == "Return Probability":
        st.markdown("##  Return Probability Prediction")
        st.markdown("""
        <div class="metric-card">
            <h3> What this does:</h3>
            <p>Predicts whether a customer will return a product based on order characteristics, 
            customer demographics, and product information. This helps with inventory management 
            and customer service planning.</p>
        </div>
        """, unsafe_allow_html=True)
        
      
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from return_probability import main as return_prob_main
        return_prob_main()

    elif st.session_state.current_page == "Return Reason":
        st.markdown("## Return Reason Analysis")
        st.markdown("""
        <div class="metric-card">
            <h3> What this does:</h3>
            <p>Analyzes the most likely reasons for product returns, helping identify patterns 
            in customer dissatisfaction and product quality issues. Provides top 3 predicted 
            return reasons with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
        
       
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from return_reason import main as return_reason_main
        return_reason_main()

    elif st.session_state.current_page == "Bracketing Detection":
        st.markdown("## Bracketing Behavior Detection")
        st.markdown("""
        <div class="metric-card">
            <h3>What this does:</h3>
            <p>Detects when customers order multiple sizes or variants of the same product 
            with the intention of keeping one and returning the rest. This helps identify 
            potential abuse of return policies.</p>
        </div>
        """, unsafe_allow_html=True)
        
   
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from bracket import main as bracket_main
        bracket_main()


st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p> Built for Sparkathon | Reimagining customer experience</p>
    <p>Powered by Streamlit, Scikit-learn, and AutoGluon</p>
</div>
""", unsafe_allow_html=True)