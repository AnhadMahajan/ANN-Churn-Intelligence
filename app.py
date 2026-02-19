import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background: #080C10 !important;
    color: #C8D6E5 !important;
}

.stApp {
    background: #080C10 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 3rem 3rem 3rem !important;
    max-width: 1100px !important;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: flex-end;
    gap: 1.25rem;
    margin-bottom: 0.25rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1C2530;
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #E8F0FA;
    line-height: 1;
    letter-spacing: -0.04em;
    margin: 0;
}
.app-title span {
    color: #3EB8FF;
}
.app-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3EB8FF;
    background: rgba(62,184,255,0.08);
    border: 1px solid rgba(62,184,255,0.2);
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    margin-bottom: 0.4rem;
}
.app-sub {
    font-size: 0.78rem;
    color: #4A6070;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3EB8FF;
    margin-bottom: 0.75rem;
    margin-top: 1.75rem;
    opacity: 0.85;
}

/* ── Card ── */
.card {
    background: #0D1520;
    border: 1px solid #1C2A38;
    border-radius: 10px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: #253545; }

/* ── Streamlit input overrides ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div.stNumberInput > div > div > input {
    background: #101C28 !important;
    border: 1px solid #1E2F40 !important;
    border-radius: 7px !important;
    color: #C8D6E5 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    transition: border-color 0.15s !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within,
div.stNumberInput > div > div > input:focus {
    border-color: #3EB8FF !important;
    box-shadow: 0 0 0 3px rgba(62,184,255,0.08) !important;
}

/* Dropdown menu */
div[data-baseweb="popover"] {
    background: #0D1B28 !important;
    border: 1px solid #1C2A38 !important;
    border-radius: 8px !important;
}
li[role="option"] {
    background: #0D1B28 !important;
    color: #C8D6E5 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}
li[role="option"]:hover {
    background: #162230 !important;
    color: #3EB8FF !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] > div > div > div > div {
    background: #3EB8FF !important;
}
div[data-testid="stSlider"] > div > div > div {
    background: #1C2A38 !important;
}

/* ── Labels ── */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 400 !important;
    color: #4A6070 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Predict button ── */
div.stButton > button {
    width: 100%;
    background: #3EB8FF !important;
    color: #050A10 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.5rem !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    margin-top: 0.5rem;
}
div.stButton > button:hover {
    background: #6ACBFF !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(62,184,255,0.25) !important;
}
div.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Result panel ── */
.result-panel {
    border-radius: 10px;
    padding: 2rem 2rem;
    margin-top: 1.5rem;
    border: 1px solid;
    animation: fadeIn 0.4s ease;
}
.result-panel.churn {
    background: linear-gradient(135deg, #1A0A0A 0%, #120810 100%);
    border-color: #FF4D4D;
}
.result-panel.safe {
    background: linear-gradient(135deg, #061512 0%, #060E14 100%);
    border-color: #00D68F;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-verdict {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.3rem;
}
.result-verdict.churn { color: #FF6B6B; }
.result-verdict.safe  { color: #00D68F; }
.result-desc {
    font-size: 0.78rem;
    color: #4A6070;
    letter-spacing: 0.04em;
}

/* ── Probability bar ── */
.prob-container {
    margin-top: 1.25rem;
}
.prob-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.45rem;
}
.prob-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4A6070;
}
.prob-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.03em;
}
.prob-value.churn { color: #FF6B6B; }
.prob-value.safe  { color: #00D68F; }
.prob-bar-track {
    height: 6px;
    background: #141E28;
    border-radius: 99px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s cubic-bezier(.16,1,.3,1);
}
.prob-bar-fill.churn { background: linear-gradient(90deg, #FF4D4D, #FF8080); }
.prob-bar-fill.safe  { background: linear-gradient(90deg, #00A870, #00D68F); }

/* ── Threshold marker ── */
.threshold-row {
    display: flex;
    justify-content: flex-end;
    margin-top: 0.3rem;
}
.threshold-badge {
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    color: #2A3D50;
    text-transform: uppercase;
}

/* ── Divider ── */
hr { border-color: #1C2530 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080C10; }
::-webkit-scrollbar-thumb { background: #1C2A38; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    m = load_model('model.h5')
    with open('artifacts/encoders/label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('artifacts/encoders/onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('artifacts/scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    return m, le_gender, ohe_geo, sc

model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">Churn<span>Intelligence</span></div>
        <div class="app-sub">Neural network-powered customer attrition analysis</div>
    </div>
    <div class="app-badge">v1.0 Live</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:

    # -- Personal Info --
    st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        with c2:
            gender = st.selectbox('Gender', label_encoder_gender.classes_)
        with c3:
            age = st.slider('Age', 18, 92, 35)

    # -- Financial --
    st.markdown('<div class="section-label">Financials</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
        with c2:
            balance = st.number_input('Balance', min_value=0.0, value=60000.0, step=500.0)
        with c3:
            estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, step=500.0)

    # -- Relationship --
    st.markdown('<div class="section-label">Relationship</div>', unsafe_allow_html=True)
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tenure = st.slider('Tenure (yrs)', 0, 10, 3)
        with c2:
            num_of_products = st.slider('Products', 1, 4, 1)
        with c3:
            has_cr_card = st.selectbox('Credit Card', options=[0, 1], format_func=lambda x: 'Yes' if x else 'No')
        with c4:
            is_active_member = st.selectbox('Active Member', options=[0, 1], format_func=lambda x: 'Yes' if x else 'No')

    st.markdown('<br>', unsafe_allow_html=True)
    predict_btn = st.button('Run Prediction')


# ── Prediction logic ──────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)

    if predict_btn:
        # Build input
        input_data = pd.DataFrame({
            'CreditScore':      [credit_score],
            'Gender':           [label_encoder_gender.transform([gender])[0]],
            'Age':              [age],
            'Tenure':           [tenure],
            'Balance':          [balance],
            'NumOfProducts':    [num_of_products],
            'HasCrCard':        [has_cr_card],
            'IsActiveMember':   [is_active_member],
            'EstimatedSalary':  [estimated_salary],
        })

        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

        input_scaled = scaler.transform(input_data)
        prob = float(model.predict(input_scaled, verbose=0)[0][0])

        is_churn = prob > 0.5
        css_cls  = "churn" if is_churn else "safe"
        verdict  = "Likely to Churn" if is_churn else "Low Churn Risk"
        desc     = (
            "This customer shows elevated attrition signals. Immediate retention action is recommended."
            if is_churn else
            "This customer appears stable. Standard engagement cadence is sufficient."
        )
        pct = f"{prob * 100:.1f}%"
        bar_width = f"{prob * 100:.1f}%"

        st.markdown(f"""
        <div class="result-panel {css_cls}">
            <div class="result-verdict {css_cls}">{verdict}</div>
            <div class="result-desc">{desc}</div>
            <div class="prob-container">
                <div class="prob-header">
                    <div class="prob-label">Churn Probability</div>
                    <div class="prob-value {css_cls}">{pct}</div>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill {css_cls}" style="width:{bar_width}"></div>
                </div>
                <div class="threshold-row">
                    <div class="threshold-badge">Threshold: 50%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Slim input summary ─────────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1.5rem">Input Summary</div>', unsafe_allow_html=True)
        summary_rows = {
            "Geography":    geography,
            "Gender":       gender,
            "Age":          age,
            "Credit Score": credit_score,
            "Balance":      f"{balance:,.0f}",
            "Salary":       f"{estimated_salary:,.0f}",
            "Tenure":       f"{tenure} yr{'s' if tenure != 1 else ''}",
            "Products":     num_of_products,
            "Credit Card":  "Yes" if has_cr_card else "No",
            "Active":       "Yes" if is_active_member else "No",
        }
        rows_html = "".join(
            f'<div style="display:flex;justify-content:space-between;padding:0.45rem 0;'
            f'border-bottom:1px solid #111D28;">'
            f'<span style="font-size:0.7rem;color:#3A5060;letter-spacing:0.06em;text-transform:uppercase">{k}</span>'
            f'<span style="font-size:0.78rem;color:#8AAFC5;font-family:\'DM Mono\',monospace">{v}</span>'
            f'</div>'
            for k, v in summary_rows.items()
        )
        st.markdown(f'<div class="card" style="margin-top:0">{rows_html}</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            margin-top: 2rem;
            padding: 2.5rem 2rem;
            border: 1px dashed #1A2A38;
            border-radius: 10px;
            text-align: center;
        ">
            <div style="
                font-family: 'Syne', sans-serif;
                font-size: 1rem;
                font-weight: 700;
                color: #1E3040;
                letter-spacing: -0.01em;
                margin-bottom: 0.5rem;
            ">Awaiting input</div>
            <div style="font-size: 0.7rem; color: #1A2A38; letter-spacing: 0.06em; text-transform: uppercase;">
                Fill in the form and run prediction
            </div>
        </div>
        """, unsafe_allow_html=True)