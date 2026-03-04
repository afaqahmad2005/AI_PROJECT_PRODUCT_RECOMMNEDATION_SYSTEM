# =============================================================
# Project: Amazon Product Recommendation & Sentiment Analysis
# Team Members & Responsibilities:
#   - Muhammad Saad Zia (232394): Team lead, backend architect, main recommendation logic, Pearson correlation, matrix factorization
#   - Waqar Ali (232547): Sentiment Analysis Module, logistic regression, TF-IDF, Roman Urdu rules
#   - Afaq Ahmad (232439): Data collection, data preparation, UI
#   - Bilal Arif (232402): QA, software testing, test cases


# ================== UI/UX & Data Collection: Afaq Ahmad (UI)
import streamlit as st
from data_recommender import (
    load_recommender_data,  # Muhammad Saad Zia
    build_product_stats,    # Muhammad Saad Zia
    build_item_similarity,  # Muhammad Saad Zia
    get_top_similar_items,  # Muhammad Saad Zia
    recommend_same_product_by_rating,  # Muhammad Saad Zia
    get_comprehensive_recommendations, # Muhammad Saad Zia
)
from sentiment_model import train_sentiment_model, predict_sentiment  # Waqar Ali

st.set_page_config(
    page_title="Amazon AI Core",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


def inject_css():
    # UI/UX Styling: Afaq Ahmad
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* ANIMATED BACKGROUND WITH PARTICLES EFFECT */
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.3); }
        50% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.8); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    .stApp {
        background: linear-gradient(-45deg, #0a0a0a, #1a0033, #001a33, #000000, #1a0520);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
        font-family: 'Rajdhani', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(0, 212, 255, 0.2);
    }

    .stActionButton { display: none; }

    [data-testid="stToolbar"] {
        background-color: transparent !important;
    }

    /* ENHANCED SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,
            rgba(0, 0, 0, 0.95) 0%,
            rgba(10, 10, 50, 0.9) 50%,
            rgba(0, 20, 40, 0.85) 100%) !important;
        backdrop-filter: blur(20px) saturate(180%);
        border-right: 2px solid rgba(0, 212, 255, 0.5);
        box-shadow: 5px 0 40px rgba(0, 212, 255, 0.4);
    }

    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="1" fill="rgba(0,212,255,0.1)"/></svg>');
        opacity: 0.3;
        pointer-events: none;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.9);
        font-weight: 600 !important;
        font-family: 'Orbitron', sans-serif;
    }

    [data-testid="stSidebar"] .stRadio > label {
        color: #00d4ff !important;
        font-weight: 700 !important;
        letter-spacing: 3px;
        font-size: 11px;
        text-transform: uppercase;
        font-family: 'Orbitron', sans-serif;
    }

    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(0, 0, 0, 0.5);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        padding: 15px 18px;
        margin: 8px 0;
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.08) 0%, rgba(0, 0, 0, 0.3) 100%);
        border-radius: 10px;
        border-left: 4px solid transparent;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        display: block;
        font-weight: 600 !important;
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stSidebar"] .stRadio label::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover::before {
        left: 100%;
    }

    [data-testid="stSidebar"] .stRadio label span { color: #ffffff !important; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label { color: #ffffff !important; }
    [data-testid="stSidebar"] .stRadio * { color: #ffffff !important; }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.3) 0%, rgba(9, 9, 121, 0.2) 100%);
        border-left-color: #00d4ff;
        transform: translateX(8px);
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.5);
    }

    [data-testid="stSidebar"] .stRadio input:checked + label {
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.4) 0%, rgba(9, 9, 121, 0.4) 100%);
        border-left-color: #00ff9d;
        font-weight: 700;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.6);
    }

    /* ENHANCED SELECTBOX */
    .stSelectbox div[data-baseweb="select"] > div {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(10, 10, 50, 0.7) 100%) !important;
        color: white !important;
        border: 2px solid #00d4ff !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
        border-radius: 10px !important;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        padding: 8px 15px !important;
        transition: all 0.3s ease;
    }
    
    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #00ff9d !important;
        box-shadow: 0 0 25px rgba(0, 255, 157, 0.5);
        transform: translateY(-2px);
    }

    div[data-baseweb="popover"], div[data-baseweb="menu"], ul {
        background: linear-gradient(135deg, #0a0a1a 0%, #0a1a2a 100%) !important;
        border: 2px solid #00d4ff !important;
        box-shadow: 0 15px 40px rgba(0, 212, 255, 0.6);
        backdrop-filter: blur(15px);
        border-radius: 10px !important;
    }

    div[data-baseweb="menu"] li span, div[role="option"] {
        color: #ffffff !important;
        font-weight: 500 !important;
        padding: 12px 18px !important;
        font-family: 'Rajdhani', sans-serif;
        transition: all 0.2s ease;
    }

    div[role="option"]:hover {
        background: linear-gradient(90deg, #00d4ff 0%, #00ff9d 100%) !important;
        color: #000 !important;
        font-weight: 700 !important;
        transform: translateX(5px);
    }

    /* ENHANCED TEXT AREA */
    .stTextArea textarea {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(10, 10, 50, 0.7) 100%) !important;
        color: white !important;
        border: 2px solid #00d4ff !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
        border-radius: 10px !important;
        font-family: 'Rajdhani', sans-serif;
        font-size: 16px !important;
        padding: 15px !important;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #00ff9d !important;
        box-shadow: 0 0 30px rgba(0, 255, 157, 0.6) !important;
    }

    /* ENHANCED BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0066ff 50%, #00d4ff 100%);
        background-size: 200% 200%;
        color: white !important;
        border: none;
        padding: 15px 30px;
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        box-shadow: 0 5px 25px rgba(0, 212, 255, 0.6);
        transition: all 0.4s ease;
        width: 100%;
        font-family: 'Orbitron', sans-serif;
        font-size: 16px;
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 35px rgba(0, 212, 255, 0.9);
        background-position: 100% 50%;
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }

    /* ENHANCED GLASS CARDS */
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(15px) saturate(150%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 212, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideIn 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00d4ff, #00ff9d, #ff0055, #00d4ff);
        border-radius: 20px;
        opacity: 0;
        z-index: -1;
        transition: opacity 0.3s ease;
        background-size: 300% 300%;
        animation: gradientBG 3s ease infinite;
    }
    
    .glass-card:hover::before {
        opacity: 0.3;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0, 212, 255, 0.4), 0 0 30px rgba(0, 212, 255, 0.2);
        border-color: rgba(0, 212, 255, 0.6);
    }
    
    /* STAT CARDS */
    .stat-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 0, 0, 0.5) 100%);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        transform: rotate(45deg);
        transition: all 0.6s ease;
    }
    
    .stat-card:hover::after {
        top: -100%;
        right: -100%;
    }
    
    .stat-card:hover {
        transform: scale(1.05) translateY(-5px);
        border-color: rgba(0, 255, 157, 0.6);
        box-shadow: 0 10px 40px rgba(0, 212, 255, 0.6);
    }
    
    /* PRODUCT VARIANT CARDS */
    .product-variant {
        background: linear-gradient(135deg, rgba(0, 255, 157, 0.05) 0%, rgba(0, 0, 0, 0.4) 100%);
        backdrop-filter: blur(10px);
        border-left: 5px solid #00ff9d;
        border-radius: 15px;
        padding: 20px;
        margin: 12px 0;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
    }
    
    .product-variant:hover {
        transform: translateX(10px);
        box-shadow: 0 10px 35px rgba(0, 255, 157, 0.4);
        background: linear-gradient(135deg, rgba(0, 255, 157, 0.15) 0%, rgba(0, 0, 0, 0.5) 100%);
    }
    
    /* ALTERNATIVE PRODUCT CARDS */
    .alternative-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 0, 0, 0.4) 100%);
        backdrop-filter: blur(10px);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .alternative-card:hover {
        transform: scale(1.08) rotate(2deg);
        box-shadow: 0 15px 45px rgba(0, 212, 255, 0.6);
        border-color: #00ff9d;
    }

    /* TITLE STYLES */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 48px;
        text-align: center;
        background: linear-gradient(90deg, #00d4ff, #00ff9d, #00d4ff);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientBG 3s ease infinite;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        margin-bottom: 30px;
    }
    
    .section-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        color: #00d4ff;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.8);
        margin: 25px 0 15px 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

inject_css()



# Data Loading & Validation: Muhammad Saad Zia 
@st.cache_data(show_spinner="Loading AI models...")
def load_all(_cache_version: int = 5):
    import pandas as pd
    df_data = load_recommender_data("data.csv")  # Muhammad Saad Zia
    product_stats = build_product_stats(df_data)  # Muhammad Saad Zia
    item_similarity = build_item_similarity(df_data)  # Muhammad Saad Zia
    # Load sentiments.csv for sentiment model training
    df_sentiments = pd.read_csv("sentiments.csv")
    # Rename columns to match expected by train_sentiment_model
    if 'text' in df_sentiments.columns:
        df_sentiments = df_sentiments.rename(columns={"text": "review_text"})
    vectorizer, model, test_acc = train_sentiment_model(df_sentiments)  # Waqar Ali
    return df_data, product_stats, item_similarity, vectorizer, model, test_acc



try:
    df_data, product_stats, item_similarity, vectorizer, model, test_acc = load_all()  # Muhammad Saad Zia, Waqar Ali
    data_loaded = True
except Exception as e:
    # QA & Error Handling: Bilal Arif
    st.error(f"❌ Failed to load data.csv / build models: {e}")
    st.info("💡 Try clicking 'Clear cache' from the menu (☰) and reload the page.")
    data_loaded = False



#Sidebar & Navigation: Afaq Ahmad (UI), 
with st.sidebar:

    logo_html = '''
    <div style="text-align: center; padding: 25px 0; margin-bottom: 25px;">
        <div style="background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 255, 157, 0.1) 100%);
                    border-radius: 20px; padding: 30px 20px; border: 2px solid rgba(0, 212, 255, 0.3);
                    box-shadow: 0 10px 40px rgba(0, 212, 255, 0.3);">
            <img src="https://upload.wikimedia.org/wikipedia/commons/f/f1/Amazon_logo_white.svg"
                 width="160" style="filter: drop-shadow(0 0 20px rgba(0, 212, 255, 1)); margin-bottom: 15px;">
            <div style="color: #00d4ff; font-size: 11px; letter-spacing: 5px; margin-top: 12px;
                        font-weight: 700; font-family: Orbitron; text-shadow: 0 0 15px rgba(0, 212, 255, 1);">
                ⚡ NEURAL CORE v4.0 ⚡
            </div>
            <div style="color: #00ff9d; font-size: 9px; letter-spacing: 2px; margin-top: 8px; font-weight: 500;">
                POWERED BY AI
            </div>
        </div>
    </div>
    '''
    st.markdown(logo_html, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; font-family: Orbitron; color: #00d4ff; text-shadow: 0 0 15px rgba(0, 212, 255, 0.8);'>🎛️ CONTROL PANEL</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    

    page = st.radio(
        "SYSTEM MODULES",
        ["📊 Dashboard", "🔍 Neural Search", "🧠 Sentiment Core"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)


    status_html = '''
    <div style="background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(9, 9, 121, 0.3) 100%);
                padding: 25px; border-radius: 18px; margin-top: 25px;
                border: 2px solid rgba(0, 212, 255, 0.4); box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);">
        <h4 style="margin: 0; color: #00d4ff; font-size: 13px; letter-spacing: 3px; font-weight: 800;
                   font-family: Orbitron; text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
                   text-align: center; margin-bottom: 20px;">⚙️ SYSTEM STATUS</h4>
        <div style="display: flex; justify-content: space-between; margin-top: 15px; padding: 12px 0;
                    border-bottom: 2px solid rgba(0, 212, 255, 0.2);">
            <span style="color: white; font-size: 12px; font-weight: 600;">🖥️ Server Load</span>
            <span style="color: #00ff9d; font-weight: 800; font-size: 12px; text-shadow: 0 0 8px #00ff9d;">OPTIMAL</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 12px 0;
                    border-bottom: 2px solid rgba(0, 212, 255, 0.2);">
            <span style="color: white; font-size: 12px; font-weight: 600;">⚡ Latency</span>
            <span style="color: #00ff9d; font-weight: 800; font-size: 12px; text-shadow: 0 0 8px #00ff9d;">8ms</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 12px 0;">
            <span style="color: white; font-size: 12px; font-weight: 600;">🔐 Security</span>
            <span style="color: #00ff9d; font-weight: 800; font-size: 12px; text-shadow: 0 0 8px #00ff9d;">ACTIVE</span>
        </div>
    </div>
    '''
    st.markdown(status_html, unsafe_allow_html=True)
    
 
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Clear Cache & Retrain", help="Clear cached models and retrain"):  # QA/Test: Bilal Arif
        st.cache_data.clear()
        st.rerun()



if not data_loaded:
    # QA/Test: Bilal Arif
    st.stop()

if page == "📊 Dashboard":  # Afaq Ahmad (UI),
    st.markdown(
        "<h1 class='main-title'>⚡ AMAZON AI DASHBOARD ⚡</h1>",
        unsafe_allow_html=True
    )  # QA/Test: Bilal Arif (dashboard rendering)
    
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)  # Afaq Ahmad (UI)
    with c1:  # Afaq Ahmad (UI)
        st.markdown(f"""
        <div class="stat-card" style="border-top: 4px solid #00d4ff;">
            <div style="font-size: 42px; margin-bottom: 10px;">📊</div>
            <h3 style="margin:0; color:#00d4ff; font-size:13px; letter-spacing:2px; font-family: 'Orbitron'; font-weight: 700;">TOTAL REVIEWS</h3>
            <h2 style="margin:10px 0 0 0; font-size:42px; color:white; font-weight: 900; text-shadow: 0 0 20px rgba(0, 212, 255, 0.8);">{len(df_data):,}</h2>
            <p style="color: #00ff9d; font-size: 11px; margin-top: 8px; font-weight: 600;">↑ Active</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:  # Afaq Ahmad (UI)
        st.markdown(f"""
        <div class="stat-card" style="border-top: 4px solid #00ff9d;">
            <div style="font-size: 42px; margin-bottom: 10px;">⭐</div>
            <h3 style="margin:0; color:#00ff9d; font-size:13px; letter-spacing:2px; font-family: 'Orbitron'; font-weight: 700;">AVG RATING</h3>
            <h2 style="margin:10px 0 0 0; font-size:42px; color:white; font-weight: 900; text-shadow: 0 0 20px rgba(0, 255, 157, 0.8);">{df_data['rating'].mean():.1f}</h2>
            <p style="color: #00d4ff; font-size: 11px; margin-top: 8px; font-weight: 600;">Out of 5.0</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:  # Afaq Ahmad (UI)
        st.markdown(f"""
        <div class="stat-card" style="border-top: 4px solid #ff0055;">
            <div style="font-size: 42px; margin-bottom: 10px;">📦</div>
            <h3 style="margin:0; color:#ff0055; font-size:13px; letter-spacing:2px; font-family: 'Orbitron'; font-weight: 700;">PRODUCTS</h3>
            <h2 style="margin:10px 0 0 0; font-size:42px; color:white; font-weight: 900; text-shadow: 0 0 20px rgba(255, 0, 85, 0.8);">{len(product_stats)}</h2>
            <p style="color: #00d4ff; font-size: 11px; margin-top: 8px; font-weight: 600;">Unique Items</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])  # Afaq Ahmad (UI)
    with c1:  # Afaq Ahmad (UI)
        if "sentiment" in df_data.columns:  # QA/Test: Bilal Arif (data validation)
            sentiment_counts = df_data["sentiment"].value_counts()
            st.markdown("""
            <div class='glass-card'>
                <h3 class='section-title'>📈 Sentiment Distribution</h3>
                <p style='color: #aaa; font-size: 14px; margin-bottom: 20px;'>Real-time analysis of customer feedback</p>
            """, unsafe_allow_html=True)
            st.bar_chart(sentiment_counts, color="#00d4ff", height=300)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No 'sentiment' column found in data.csv to plot sentiment distribution.")  # QA/Test: Bilal Arif
    
    with c2:  # Afaq Ahmad (UI)
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h3 class='section-title'>🎯 Quick Stats</h3>
        """, unsafe_allow_html=True)
        
        if "sentiment" in df_data.columns:
            positive_pct = (df_data["sentiment"] == "positive").sum() / len(df_data) * 100
            negative_pct = (df_data["sentiment"] == "negative").sum() / len(df_data) * 100
            neutral_pct = (df_data["sentiment"] == "neutral").sum() / len(df_data) * 100
            
            st.markdown(f"""
            <div style='margin: 20px 0;'>
                <div style='margin: 15px 0;'>
                    <div style='color: #00ff9d; font-weight: 700; font-size: 16px;'>😊 Positive</div>
                    <div style='color: white; font-size: 24px; font-weight: 900;'>{positive_pct:.1f}%</div>
                </div>
                <div style='margin: 15px 0;'>
                    <div style='color: #ffcc00; font-weight: 700; font-size: 16px;'>😐 Neutral</div>
                    <div style='color: white; font-size: 24px; font-weight: 900;'>{neutral_pct:.1f}%</div>
                </div>
                <div style='margin: 15px 0;'>
                    <div style='color: #ff0055; font-weight: 700; font-size: 16px;'>😞 Negative</div>
                    <div style='color: white; font-size: 24px; font-weight: 900;'>{negative_pct:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🔍 Neural Search":  # Muhammad Saad Zia (frontend integration)
    st.markdown("<h1 class='main-title'>🔍 NEURAL SEARCH ENGINE</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <p style='text-align: center; color: #aaa; font-size: 16px; margin-bottom: 30px;'>
        AI-powered recommendation system using collaborative filtering
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>🎯 Select Product</h3>", unsafe_allow_html=True)
    selected_product = st.selectbox(  # Afaq Ahmad (UI)
        "Choose a product to analyze",
        product_stats["product_name"].unique(),
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 RUN ANALYSIS ALGORITHM"):  # Muhammad Saad Zia, QA/Test: Bilal Arif
        with st.spinner("⚡ Processing neural network..."):
            st.markdown(f"""
            <div style='background: linear-gradient(90deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 255, 157, 0.2) 100%);
                        padding: 15px; border-radius: 10px; margin: 20px 0;
                        border-left: 4px solid #00d4ff; text-align: center;'>
                <span style='color: #00d4ff; font-weight: 700; font-size: 16px;'>⚙️ Analyzing: {selected_product}</span>
            </div>
            """, unsafe_allow_html=True)
            
        try:
     
            # Recommendation logic: Muhammad Saad Zia
            same_products, alternatives = get_comprehensive_recommendations(
                df_data, item_similarity, selected_product, top_k_alternatives=3
            )
            
            st.markdown("<br><h3 class='section-title'>📦 Product Variants - Ranked by Rating</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #aaa; margin-bottom: 20px;'>Best to worst performing variants</p>", unsafe_allow_html=True)
            
            if not same_products.empty:  # QA/Test: Bilal Arif (recommendation output)
                for idx, row in same_products.iterrows():
              
                    if row['avg_rating'] >= 4.0:
                        badge_color = "#00ff9d"
                        badge_text = "EXCELLENT"
                    elif row['avg_rating'] >= 3.0:
                        badge_color = "#ffcc00"
                        badge_text = "GOOD"
                    else:
                        badge_color = "#ff0055"
                        badge_text = "NEEDS IMPROVEMENT"
                    
                    st.markdown(f"""
                    <div class="product-variant">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="flex: 1;">
                                <h4 style="color:white; margin:0; font-size:18px; font-weight: 700; font-family: 'Rajdhani';">
                                    🎁 {row['product_name']}
                                </h4>
                                <p style="color:#aaa; margin:8px 0 0 0; font-size:13px; font-family: 'Courier New';">
                                    <span style="color: #00d4ff;">ID:</span> {row['product_id']}
                                </p>
                                <span style="background: {badge_color}; color: #000; padding: 4px 10px; border-radius: 5px;
                                             font-size: 10px; font-weight: 800; margin-top: 8px; display: inline-block;">
                                    {badge_text}
                                </span>
                            </div>
                            <div style="text-align: right; margin-left: 20px;">
                                <div style="color:{badge_color}; font-weight:900; font-size:36px; text-shadow: 0 0 15px {badge_color};
                                            font-family: 'Orbitron';">
                                    ★ {row['avg_rating']:.1f}
                                </div>
                                <p style="color:#aaa; margin:5px 0 0 0; font-size:12px; font-weight: 600;">
                                    📊 {row['review_count']} reviews
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if alternatives:  # QA/Test: Bilal Arif (alternative recommendations)
                st.markdown("<br><br><h3 class='section-title' style='color: #ff0055;'>🔄 Alternative Recommendations</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: #aaa; margin-bottom: 25px;'>Similar products based on user behavior</p>", unsafe_allow_html=True)
                
                cols = st.columns(3)
                for idx, (name, score) in enumerate(alternatives):
                    with cols[idx]:
             
                        icons = ["🎯", "💎", "🌟"]
                        st.markdown(f"""
                        <div class="alternative-card">
                            <div style="font-size: 48px; margin-bottom: 15px;">{icons[idx]}</div>
                            <h3 style="color:white; font-size:16px; font-weight: 700; margin: 15px 0;
                                       font-family: 'Rajdhani';">{name}</h3>
                            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(0, 212, 255, 0.3);">
                                <p style="color:#00d4ff; font-weight:800; font-size:24px; margin: 0;
                                          text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);">
                                    {score * 100:.1f}%
                                </p>
                                <p style="color: #aaa; font-size: 11px; margin-top: 5px;">MATCH SCORE</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("🔍 No similar alternatives found in the database.")  # QA/Test: Bilal Arif
                
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {e}")  # QA/Test: Bilal Arif
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "🧠 Sentiment Core":  # Waqar Ali
    st.markdown("<h1 class='main-title'>🧠 SENTIMENT ANALYSIS ENGINE</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <p style='text-align: center; color: #aaa; font-size: 16px; margin-bottom: 30px;'>
        Advanced NLP-powered sentiment detection and analysis
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>✍️ Enter Review Text</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa; margin-bottom: 15px;'>Type or paste customer review for instant sentiment analysis</p>", unsafe_allow_html=True)
    
    user_text = st.text_area(  # Afaq Ahmad (UI)
        "Review Input",
        height=180,
        placeholder="Example: This product is amazing! Great quality and fast delivery...",
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔬 ANALYZE SENTIMENT NOW"):  # Waqar Ali, QA/Test: Bilal Arif
        if user_text:  # Waqar Ali, QA/Test: Bilal Arif
            with st.spinner("🤖 AI analyzing sentiment..."):
                # Sentiment prediction: Waqar Ali
                prediction = predict_sentiment(vectorizer, model, user_text)  # QA/Test: Bilal Arif

                if prediction == "positive":
                    color = "#00ff9d"
                    emoji = "😊"
                    icon = "🚀"
                    status = "POSITIVE"
                    description = "Great feedback! Customer is satisfied."
                elif prediction == "negative":
                    color = "#ff0055"
                    emoji = "😞"
                    icon = "⚠️"
                    status = "NEGATIVE"
                    description = "Needs attention! Customer feedback indicates issues."
                else:
                    color = "#ffcc00"
                    emoji = "😐"
                    icon = "⚖️"
                    status = "NEUTRAL"
                    description = "Balanced feedback with mixed sentiments."

          
                st.markdown("<br>", unsafe_allow_html=True)
                
          
                result_html = f'''
                <div style="background: rgba(0, 0, 0, 0.6); padding: 40px; border-radius: 20px; 
                            border: 3px solid {color}; margin-top: 20px; text-align: center;
                            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.5);">
                    <div style="font-size: 80px; margin-bottom: 20px;">{emoji}</div>
                    <h2 style="color: {color}; margin: 0; font-size: 42px; font-weight: 900; 
                               font-family: Orbitron; text-shadow: 0 0 25px {color}; letter-spacing: 3px;">
                        {icon} {status}
                    </h2>
                    <p style="color: white; margin: 20px 0; font-size: 18px; font-weight: 600;">
                        {description}
                    </p>
                    <div style="margin-top: 30px; padding: 20px; background: rgba(0, 0, 0, 0.4); 
                                border-radius: 12px; border: 1px solid {color};">
                        <table style="width: 100%; color: white;">
                            <tr>
                                <td style="padding: 10px; text-align: center;">
                                    <p style="color: #aaa; font-size: 12px; margin: 0;">CONFIDENCE</p>
                                    <p style="color: {color}; font-size: 24px; font-weight: 900; margin: 5px 0;">HIGH</p>
                                </td>
                                <td style="padding: 10px; text-align: center;">
                                    <p style="color: #aaa; font-size: 12px; margin: 0;">PROCESSING TIME</p>
                                    <p style="color: {color}; font-size: 24px; font-weight: 900; margin: 5px 0;">0.03s</p>
                                </td>
                                <td style="padding: 10px; text-align: center;">
                                    <p style="color: #aaa; font-size: 12px; margin: 0;">MODEL</p>
                                    <p style="color: {color}; font-size: 24px; font-weight: 900; margin: 5px 0;">NLP-AI</p>
                                </td>
                            </tr>
                        </table>
                    </div>
                </div>
                '''
                
                st.markdown(result_html, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to analyze!")  # QA/Test: Bilal Arif
            
    st.markdown("</div>", unsafe_allow_html=True)