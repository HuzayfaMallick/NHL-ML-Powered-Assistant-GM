import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="NHL AI-GM", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    body {
        background-color: #0A192F;
    }
    .welcome-container {
        text-align: center;
        padding-top: 180px;
        font-family: 'Montserrat', sans-serif;
    }
    h1 {
        font-size: 55px;
        color: #F8FAFC;
        animation: fadeIn 2s ease-in;
    }
    .subtitle {
        font-size: 20px;
        color: #94A3B8;
        margin-top: 10px;
        animation: fadeIn 3s ease-in;
    }
    .stButton button {
        background-color: #FF4C29;
        color: white;
        font-size: 18px;
        padding: 12px 30px;
        border-radius: 30px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #ff6a4d;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)

# --- WELCOME CONTENT ---
st.markdown('<div class="welcome-container"><h1>Welcome to NHL AI-GM</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your advanced analytics assistant for smarter hockey decisions</p>', unsafe_allow_html=True)

# --- CONTINUE BUTTON ---
if st.button("Continue"):
    st.switch_page("pages/frontend_ML.py")