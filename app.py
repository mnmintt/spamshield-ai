import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from PIL import Image
import easyocr

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

st.set_page_config(
    page_title="SpamShield AI",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

    html, body, .stApp {
        font-family: 'Inter', sans-serif !important;
        font-size: 19px;
    }
    .material-icons,
    .material-symbols-rounded,
    .material-symbols-outlined,
    [class*="material-icons"],
    [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
        font-feature-settings: 'liga' !important;
    }

    /* Icon fallbacks: hide broken icon words and draw real symbols instead. */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneIcon"],
    [data-testid="stFileUploader"] button [class*="material-symbols"],
    [data-testid="stFileUploader"] button [class*="material-icons"],
    [data-testid="stSidebar"] button [class*="material-symbols"],
    [data-testid="stSidebar"] button [class*="material-icons"],
    [data-testid="collapsedControl"] button [class*="material-symbols"],
    [data-testid="collapsedControl"] button [class*="material-icons"] {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        font-size: 0 !important;
        line-height: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stFileUploader"] button {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    [data-testid="stFileUploader"] button::before {
        content: "↑";
        font-family: Arial, sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #8890a8 !important;
        line-height: 1 !important;
    }
    [data-testid="collapsedControl"] button {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="collapsedControl"] button::before {
        content: "«";
        font-family: Arial, sans-serif !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #c5c8dc !important;
        line-height: 1 !important;
    }

    /* ── Base ── */
    .stApp { background: #0a0a0f; color: #e2e4ee; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #07070c;
        border-right: 1px solid #1a1a2e;
    }
    [data-testid="stSidebar"] * { color: #8890a8 !important; }
    [data-testid="stSidebar"] hr { border-color: #1a1a2e !important; }
    [data-testid="stSidebar"] .stRadio label {
        color: #5a6080 !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        padding: 2px 0 !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover { color: #e2e4ee !important; }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: #12121f;
        border: 1px solid #1e1e35;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        color: #4a5068 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #e2e4ee !important;
    }
    [data-testid="stSidebar"] [data-testid="metric-container"] {
        background: #0d0d18 !important;
        border-color: #151525 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #c5c8dc !important; }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #1a0a3d 0%, #2d1060 50%, #1a0a3d 100%);
        border: 1px solid #3d1e7a;
        border-radius: 18px;
        padding: 48px 52px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 240px; height: 240px;
        background: rgba(139,92,246,0.08);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -80px; left: 30%;
        width: 300px; height: 300px;
        background: rgba(109,40,217,0.05);
        border-radius: 50%;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(139,92,246,0.2);
        border: 1px solid rgba(139,92,246,0.4);
        color: #a78bfa;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        padding: 5px 12px;
        border-radius: 99px;
        margin-bottom: 16px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 12px 0;
        letter-spacing: -0.04em;
        line-height: 1.1;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: rgba(165,140,250,0.65);
        margin: 0;
        line-height: 1.6;
        font-weight: 400;
        max-width: 520px;
    }

    /* ── Page header ── */
    .page-header {
        background: #12121f;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 24px;
        border-left: 4px solid #7c3aed;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }
    .page-header-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e4ee;
        margin: 0 0 4px 0;
        letter-spacing: -0.02em;
    }
    .page-header-sub {
        font-size: 0.8rem;
        color: #4a5068;
        margin: 0;
    }

    /* ── Result boxes ── */
    .result-spam {
        background: #1a0808;
        border: 1.5px solid #7f1d1d;
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 22px 24px;
        margin: 14px 0;
        box-shadow: 0 4px 20px rgba(239,68,68,0.15);
    }
    .result-ham {
        background: #081a0e;
        border: 1.5px solid #14532d;
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        padding: 22px 24px;
        margin: 14px 0;
        box-shadow: 0 4px 20px rgba(34,197,94,0.15);
    }
    .result-eyebrow {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #4a5068;
        margin: 0 0 6px 0;
    }
    .result-verdict {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 4px 0;
    }
    .result-spam .result-verdict { color: #ef4444; }
    .result-ham  .result-verdict { color: #22c55e; }
    .result-meta { font-size: 0.78rem; color: #4a5068; margin: 0; }

    /* ── Meter ── */
    .meter-card {
        background: #12121f;
        border: 1px solid #1e1e35;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 10px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }
    .meter-label {
        font-size: 0.65rem;
        color: #3a3f5c;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    .meter-track { background: #1e1e35; border-radius: 99px; height: 8px; overflow: hidden; }
    .meter-fill-spam { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #f97316, #ef4444); }
    .meter-fill-ham  { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #34d399, #22c55e); }
    .meter-val-spam { font-size: 1.3rem; font-weight: 800; color: #ef4444; margin-top: 8px; font-family: 'Fira Code', monospace; }
    .meter-val-ham  { font-size: 1.3rem; font-weight: 800; color: #22c55e; margin-top: 8px; font-family: 'Fira Code', monospace; }

    /* ── Keyword badges ── */
    .kw-badge {
        display: inline-block;
        background: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.35);
        color: #fca5a5;
        border-radius: 6px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'Fira Code', monospace;
    }

    /* ── History ── */
    .hist-item {
        background: #111114;
        border: 1px solid #181820;
        border-radius: 14px;
        padding: 24px 28px;
        margin: 22px 0;
        border-left: 4px solid #ef4444;
        box-shadow: none;
    }
    .hist-item.ham { border-left-color: #22c55e; }
    .hist-msg  { font-size: 1rem; color: #e2e4ee; font-weight: 700; }
    .hist-meta { font-size: 0.86rem; color: #6f7487; margin-top: 10px; font-family: 'Fira Code', monospace; letter-spacing: 0.02em; }
    .history-count { font-size: 0.86rem; color: #6f7487; margin: 14px 0 4px 0; font-weight: 600; }

    /* ── Feature cards ── */
    .feat-card {
        background: #12121f;
        border-radius: 14px;
        padding: 22px;
        height: 100%;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        border: 1px solid #1e1e35;
        transition: box-shadow 0.2s, transform 0.2s, border-color 0.2s;
    }
    .feat-card:hover {
        box-shadow: 0 8px 28px rgba(124,58,237,0.2);
        transform: translateY(-2px);
        border-color: #4c1d95;
    }
    .feat-tag  { font-size: 0.65rem; font-weight: 700; color: #7c3aed; text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 10px 0; font-family: 'Fira Code', monospace; }
    .feat-name { font-size: 0.95rem; font-weight: 700; color: #e2e4ee; margin: 0 0 6px 0; }
    .feat-desc { font-size: 0.78rem; color: #4a5068; margin: 0; line-height: 1.6; }

    /* ── Upload area ── */
    .upload-zone {
        background: #0d0d18;
        border: 2px dashed #1e1e35;
        border-radius: 14px;
        padding: 32px;
        text-align: center;
        margin-bottom: 16px;
    }
    .upload-title { font-size: 0.9rem; font-weight: 600; color: #4a5068; margin: 0 0 4px 0; }
    .upload-sub   { font-size: 0.75rem; color: #2d3147; margin: 0; font-family: 'Fira Code', monospace; }

    /* ── Info card ── */
    .info-card {
        background: #12121f;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        border: 1px solid #1e1e35;
    }

    /* ── Section label ── */
    .sec-label {
        font-size: 0.65rem;
        font-weight: 700;
        color: #2d3147;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 24px 0 10px 0;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #5b21b6, #7c3aed) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        padding: 9px 20px !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 2px 12px rgba(124,58,237,0.35) !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
        box-shadow: 0 4px 20px rgba(124,58,237,0.5) !important;
        transform: none !important;
    }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] > button {
        background: #12121f !important;
        color: #7c3aed !important;
        border: 1.5px solid #2d1060 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        box-shadow: none !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        border-color: #7c3aed !important;
        background: #1a0a3d !important;
    }

    /* ── Inputs ── */
    .stTextArea textarea {
        background: #0d0d18 !important;
        border: 1.5px solid #1e1e35 !important;
        border-radius: 10px !important;
        color: #e2e4ee !important;
        font-size: 0.875rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
    }
    .stTextArea textarea::placeholder { color: #2d3147 !important; }
    .stTextInput > div > div > input {
        background: #0d0d18 !important;
        border: 1.5px solid #1e1e35 !important;
        border-radius: 8px !important;
        color: #e2e4ee !important;
        font-size: 0.875rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: #0d0d18 !important;
        border: 1.5px solid #1e1e35 !important;
        border-radius: 8px !important;
        color: #e2e4ee !important;
        font-size: 0.875rem !important;
    }
    .stSelectbox svg { color: #4a5068 !important; fill: #4a5068 !important; }
    [data-baseweb="popover"] { background: #12121f !important; border: 1px solid #1e1e35 !important; }
    [data-baseweb="menu"] { background: #12121f !important; }
    [role="option"] { background: #12121f !important; color: #c5c8dc !important; }
    [role="option"]:hover { background: #1a1a2e !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d0d18 !important;
        border: 1px solid #1e1e35 !important;
        border-bottom: 2px solid #1e1e35 !important;
        border-radius: 12px !important;
        gap: 10px !important;
        padding: 8px 10px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.25) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #aab1c8 !important;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border-radius: 10px !important;
        padding: 14px 26px !important;
        border: 1px solid transparent !important;
        border-bottom: 2px solid transparent !important;
        margin-bottom: -10px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #171727 !important;
        color: #e2e4ee !important;
        border-color: #2c2c42 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #24242f !important;
        color: #ffffff !important;
        border-color: #3a3a4c !important;
        border-bottom: 3px solid #ef4444 !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border: 1px solid #1e1e35 !important;
        border-radius: 10px !important;
        background: #12121f !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.2) !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #c5c8dc !important;
    }
    [data-testid="stExpander"] p { color: #8890a8 !important; }

    /* ── Alerts ── */
    .stAlert {
        border-radius: 10px !important;
        font-size: 0.85rem !important;
        background: #12121f !important;
        border: 1px solid #1e1e35 !important;
    }
    .stAlert p { color: #c5c8dc !important; }

    /* ── Captions ── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #3a3f5c !important;
        font-size: 0.72rem !important;
        font-family: 'Fira Code', monospace !important;
    }

    /* ── Code ── */
    .stCode, [data-testid="stCode"] {
        background: #0d0d18 !important;
        border: 1px solid #1e1e35 !important;
        border-radius: 8px !important;
    }
    .stCode code { color: #a78bfa !important; }

    /* ── Markdown tables ── */
    .stMarkdown table { border-collapse: collapse; width: 100%; }
    .stMarkdown th {
        background: #0d0d18 !important; color: #4a5068 !important;
        padding: 10px 14px; border: 1px solid #1e1e35;
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700;
    }
    .stMarkdown td {
        background: #12121f !important; color: #c5c8dc !important;
        padding: 10px 14px; border: 1px solid #1a1a2e;
        font-size: 0.875rem;
    }
    .stMarkdown tr:hover td { background: #16162a !important; }

    /* ── Dataframe ── */
    .stDataFrame { border: 1px solid #1e1e35 !important; border-radius: 10px !important; }

    /* ── Radio ── */
    div[data-testid="stRadio"] label { color: #8890a8 !important; font-size: 0.875rem !important; font-weight: 500 !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #0d0d18 !important;
        border: 1.5px dashed #1e1e35 !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span { color: #3a3f5c !important; }

    /* ── Progress ── */
    .stProgress > div > div { background: linear-gradient(90deg, #5b21b6, #7c3aed) !important; }
    .stProgress > div { background: #1e1e35 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0a0a0f; }
    ::-webkit-scrollbar-thumb { background: #1e1e35; border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: #2d3147; }

    /* ── General text ── */
    p, li, span, .stMarkdown p { color: #8890a8; }
    h1, h2, h3, h4, h5, h6 { color: #e2e4ee !important; }
    strong { color: #c5c8dc !important; }

    hr { border-color: #1e1e35 !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

</style>
""", unsafe_allow_html=True)

# ── Session State
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Load Model
@st.cache_resource
def load_model():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

@st.cache_resource
def load_bert():
    from transformers import pipeline
    return pipeline("text-classification",
        model="mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection")

# ── Load Data
@st.cache_data
def load_data():
    return pd.read_csv('data/spam_cleaned.csv')

df = load_data()

# ── Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

SPAM_KEYWORDS = [
    'free', 'win', 'winner', 'claim', 'prize', 'urgent', 'click',
    'offer', 'congratulation', 'cash', 'discount', 'deal', 'limited',
    'call', 'text', 'reply', 'mobile', 'money', 'credit', 'loan',
    'guarantee', 'selected', 'reward', 'bonus', 'exclusive', 'expire',
    'subscribe', 'cancel', 'verify', 'password', 'account', 'bank'
]

def bert_label_to_prediction(label):
    label = str(label).strip().upper()
    if "SPAM" in label or label in {"LABEL_1", "1"}:
        return 1
    if "HAM" in label or "SAFE" in label or label in {"LABEL_0", "0"}:
        return 0
    return 0

def split_for_bert(text, max_words=90):
    words = str(text).split()
    if not words:
        return [str(text)[:512]]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def run_bert_prediction(text):
    try:
        classifier = load_bert()
    except Exception as e:
        st.error(f"Could not load DistilBERT. Make sure transformers, torch, and internet/model files are available. Details: {e}")
        st.stop()

    chunks = split_for_bert(text)
    bert_results = []
    for chunk in chunks:
        if chunk.strip():
            try:
                bert_results.append(classifier(chunk, truncation=True)[0])
            except Exception as e:
                st.error(f"DistilBERT prediction failed. Details: {e}")
                st.stop()

    if not bert_results:
        bert_results = [{"label": "LABEL_0", "score": 0.0}]

    spam_results = [r for r in bert_results if bert_label_to_prediction(r.get("label")) == 1]
    chosen = max(spam_results, key=lambda r: r.get("score", 0.0)) if spam_results else max(bert_results, key=lambda r: r.get("score", 0.0))
    prediction = bert_label_to_prediction(chosen.get("label"))
    confidence = float(chosen.get("score", 0.0))
    cleaned = preprocess_text(text)
    found_keywords = [w for w in SPAM_KEYWORDS if w in cleaned.lower()]
    return prediction, confidence, cleaned, found_keywords

def get_confidence(dv):
    return float(1 / (1 + np.exp(-abs(dv))))

def run_prediction(text, sel_model):
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = sel_model.predict(vectorized)[0]
    decision = sel_model.decision_function(vectorized)[0]
    confidence = get_confidence(decision)
    found_keywords = [w for w in SPAM_KEYWORDS if w in cleaned.lower()]
    return prediction, confidence, cleaned, found_keywords

def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name.lower()
    text = ""
    try:
        if filename.endswith('.txt'):
            text = uploaded_file.read().decode('utf-8', errors='ignore')
        elif filename.endswith('.pdf'):
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif filename.endswith('.docx'):
            import docx
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            text = uploaded_file.read().decode('utf-8', errors='ignore')
    except Exception as e:
        st.error(f"Could not read file: {e}")
    return text.strip()

def render_result(prediction, confidence, found_keywords, source_label=""):
    pct = int(confidence * 100)
    if prediction == 1:
        st.markdown(f"""
        <div class="result-spam">
            <p class="result-eyebrow">Detection Result{' — ' + source_label if source_label else ''}</p>
            <p class="result-verdict">Spam Detected</p>
            <p class="result-meta">Confidence: {confidence:.1%} &nbsp;&middot;&nbsp; {len(found_keywords)} trigger words found</p>
        </div>""", unsafe_allow_html=True)
        fill, val = "meter-fill-spam", "meter-val-spam"
    else:
        st.markdown(f"""
        <div class="result-ham">
            <p class="result-eyebrow">Detection Result{' — ' + source_label if source_label else ''}</p>
            <p class="result-verdict">Safe Message</p>
            <p class="result-meta">Confidence: {confidence:.1%} &nbsp;&middot;&nbsp; {len(found_keywords)} trigger words found</p>
        </div>""", unsafe_allow_html=True)
        fill, val = "meter-fill-ham", "meter-val-ham"

    st.markdown(f"""
    <div class="meter-card">
        <div class="meter-label">Confidence Score</div>
        <div class="meter-track"><div class="{fill}" style="width:{pct}%"></div></div>
        <div class="{val}">{pct}%</div>
    </div>""", unsafe_allow_html=True)

    if found_keywords:
        st.markdown('<p class="sec-label" style="margin-top:14px">Trigger Words</p>', unsafe_allow_html=True)
        badges = "".join([f'<span class="kw-badge">{w}</span>' for w in found_keywords])
        st.markdown(badges, unsafe_allow_html=True)

# ── Sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 0 20px 0'>
        <div style='font-size:0.6rem;font-weight:700;color:#7c3aed;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:8px'>SpamShield AI</div>
        <div style='font-size:1.2rem;font-weight:800;color:#e2e4ee;letter-spacing:-0.03em;line-height:1.25'>Spam Detection<br>System</div>
        <div style='margin-top:10px;display:inline-block;background:#1a0a3d;border:1px solid #3d1e7a;border-radius:6px;padding:3px 10px'>
            <span style='font-size:0.75rem;color:#6d40c4;font-family:"Fira Code",monospace'>BY GROUP SADDAM</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("", [
        "Home", "Text Analyzer", "File Scanner",
        "Batch Prediction", "Image Scanner",
        "Data Explorer", "Visualizations",
        "Model Info", "History"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p style="font-size:0.6rem;font-weight:700;color:#2d3147;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:10px">Session Stats</p>', unsafe_allow_html=True)

    total = len(st.session_state.history)
    spam_count = sum(1 for h in st.session_state.history if h['result'] == 'SPAM')
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Analyzed", total)
        st.metric("Spam", spam_count)
    with c2:
        st.metric("Safe", total - spam_count)
        st.metric("Rate", f"{(spam_count/total*100):.0f}%" if total > 0 else "—")

    if total > 0:
        if st.button("Clear session", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">NLP · Machine Learning</div>
        <p class="hero-title">SpamShield AI</p>
        <p class="hero-sub">Automatically detect spam in messages, files, and images using natural language processing and machine learning — trained on 5,572 real messages.</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Training Data", f"{len(df):,}")
    with c2: st.metric("Spam Samples", f"{df['label'].value_counts().get('spam', 0):,}")
    with c3: st.metric("Ham Samples", f"{df['label'].value_counts().get('ham', 0):,}")
    with c4: st.metric("Models Trained", "4")

    st.markdown('<p class="sec-label" style="margin-top:32px">What you can do</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    features = [
        ("TEXT", "Text Analyzer", "Type or paste any message and get an instant spam verdict with confidence score and trigger word breakdown"),
        ("FILE", "File Scanner", "Upload a .txt, .pdf, or .docx file and scan its full content for spam indicators"),
        ("IMAGE", "Image Scanner", "Upload an image of a message and use OCR to extract and analyze the text for spam"),
        ("BATCH", "Batch Prediction", "Upload a CSV with many messages and analyze all of them at once with downloadable results"),
        ("CHART", "Visualizations", "Explore word clouds, frequency charts, confusion matrix, and model performance comparisons"),
        ("LOG", "History", "Review and export every message analyzed during this session as a CSV file"),
    ]
    for col, (tag, name, desc) in zip([c1, c2, c3, c4, c5, c6], features):
        with col:
            st.markdown(f"""
            <div class="feat-card" style="margin-bottom:16px">
                <p class="feat-tag">{tag}</p>
                <p class="feat-name">{name}</p>
                <p class="feat-desc">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="sec-label">About this project</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <p style='font-size:0.875rem;color:#8890a8;line-height:1.75;margin:0'>
            SpamShield AI is a university NLP project that classifies messages as spam or legitimate.
            Four models were trained and evaluated — Naive Bayes, SVM with TF-IDF, SVM with Word2Vec,
            and <strong style="color:#a78bfa">BERT Transformer</strong> (advanced NLP).
            The best performing traditional model, <strong style="color:#a78bfa">SVM with TF-IDF</strong>,
            achieved 98.3% accuracy and is deployed alongside BERT for comparison.
            </p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="sec-label">Team members</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <div style='font-size:0.875rem;color:#e2e4ee;font-weight:600;margin-bottom:10px'>Chan Zi Chao <span style='color:#4a5068;font-weight:400'>— Data & NLP Pipeline</span></div>
            <div style='font-size:0.875rem;color:#e2e4ee;font-weight:600;margin-bottom:10px'>Muhammad Ilham <span style='color:#4a5068;font-weight:400'>— Model Training & Evaluation</span></div>
            <div style='font-size:0.875rem;color:#e2e4ee;font-weight:600;margin-bottom:10px'>Muhammad Iskandar Zulkarnain <span style='color:#4a5068;font-weight:400'>— Streamlit Application</span></div>
            <div style='font-size:0.875rem;color:#e2e4ee;font-weight:600'>Tey Yu Yang <span style='color:#4a5068;font-weight:400'>— Visualizations & Documentation</span></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — TEXT ANALYZER
# ══════════════════════════════════════════════════════════
elif page == "Text Analyzer":
    st.markdown("""
    <div class="page-header">
        <p class="page-header-title">Text Analyzer</p>
        <p class="page-header-sub">Paste any message to instantly detect spam</p>
    </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        user_input = st.text_area("Message", height=180,
            placeholder="Paste your email or SMS here...\n\nExample : Congratulations! You've won a FREE prize. Click here to claim now!",
            key="text_input", label_visibility="collapsed")

        char_count = len(user_input)
        word_count = len(user_input.split()) if user_input else 0
        ca, cb, cc = st.columns(3)
        with ca: st.caption(f"{char_count} chars")
        with cb: st.caption(f"{word_count} words")
        with cc: st.caption("Above avg length" if char_count > 138 else "Normal length")
        model_choice = st.selectbox("Select Model", [
        "SVM + TF-IDF",
        "DistilBERT Transformer"
        ])
        b1, b2 = st.columns([4, 1])
        with b1: analyze_btn = st.button("Analyze", use_container_width=True)
        with b2:
            if st.button("Clear", use_container_width=True): st.rerun()

        st.markdown('<p class="sec-label" style="margin-top:16px">Try these examples</p>', unsafe_allow_html=True)
        t1, t2 = st.tabs(["Spam", "Safe"])
        with t1: st.code("Free entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121", language=None)
        with t2: st.code("Hey, are you coming to the meeting tomorrow at 10am?", language=None)

    with col_right:
        if analyze_btn:
            if not user_input.strip():
                st.warning("Please enter a message first.")
            else:
                with st.spinner("Analyzing..."):
                    if model_choice == "DistilBERT Transformer":
                        prediction, confidence, cleaned, found_keywords = run_bert_prediction(user_input)
                    else:
                        prediction, confidence, cleaned, found_keywords = run_prediction(user_input, model)

                render_result(prediction, confidence, found_keywords)
                with st.expander("Preprocessing Details"):
                    st.markdown("**After cleaning:**")
                    st.code(cleaned, language=None)
                st.session_state.history.append({
                    'message': user_input[:55] + ('...' if len(user_input) > 55 else ''),
                    'result': 'SPAM' if prediction == 1 else 'SAFE',
                    'confidence': f"{confidence:.1%}",
                    'keywords': len(found_keywords),
                    'source': 'Text',
                    'model': model_choice,
                    
            })
        else:
            st.markdown("""
            <div style="background:#0d0d18;border:1.5px dashed #1e1e35;border-radius:12px;padding:44px;text-align:center">
                <p style="font-size:0.85rem;color:#2d3147;margin:0">Enter a message and click <strong style="color:#7c3aed">Analyze</strong></p>
            </div>""", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        st.markdown('<p class="sec-label">Recent</p>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-6:]):
            css = "hist-item" if item['result'] == 'SPAM' else "hist-item ham"
            st.markdown(f"""
            <div class="{css}">
                <div class="hist-msg"><strong>{item['result']}</strong> — {item['message']}</div>
                <div class="hist-meta">{item['confidence']} &middot; {item['keywords']} trigger words &middot; {item['source']} &middot; Model: {item.get('model', 'SVM + TF-IDF')} </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — FILE SCANNER
# ════════════════════════════════════
elif page == "File Scanner":
    st.markdown("""
    <div class="page-header">
        <p class="page-header-title">File Scanner</p>
        <p class="page-header-sub">Upload a file and scan its content for spam</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-zone">
        <p class="upload-title">Drop a file here to scan</p>
        <p class="upload-sub">.txt &nbsp;&middot;&nbsp; .pdf &nbsp;&middot;&nbsp; .docx</p>
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=['txt', 'pdf', 'docx'], label_visibility="collapsed")
    if uploaded_file:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            extracted_text = extract_text_from_file(uploaded_file)
        if not extracted_text:
            st.error("Could not extract text from this file.")
        else:
            st.success(f"Extracted {len(extracted_text):,} characters from {uploaded_file.name}")
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown('<p class="sec-label">Content Preview</p>', unsafe_allow_html=True)
                st.text_area("", value=extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""),
                             height=180, disabled=True, label_visibility="collapsed")
            with col2:
                st.markdown('<p class="sec-label">File Stats</p>', unsafe_allow_html=True)
                st.metric("Characters", f"{len(extracted_text):,}")
                st.metric("Words", f"{len(extracted_text.split()):,}")
                st.metric("Lines", f"{len(extracted_text.splitlines()):,}")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            file_model_choice = st.selectbox("Select Model", [
            "SVM + TF-IDF",
            "DistilBERT Transformer"
        ], key="file_model")

            if st.button("Scan for spam", use_container_width=True):
                with st.spinner("Scanning..."):
                    if file_model_choice == "DistilBERT Transformer":
                        prediction, confidence, cleaned, found_keywords = run_bert_prediction(extracted_text)
                    else:
                        prediction, confidence, cleaned, found_keywords = run_prediction(extracted_text, model)
                    render_result(prediction, confidence, found_keywords, source_label=uploaded_file.name)
                    st.session_state.history.append({
                        'message': f"[FILE] {uploaded_file.name}",
                        'result': 'SPAM' if prediction == 1 else 'SAFE',
                        'confidence': f"{confidence:.1%}",
                        'keywords': len(found_keywords),
                        'source': 'File',
                        'model': file_model_choice,
                    })

# ══════════════════════════════════════════════════════════
# PAGE 4 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════
elif page == "Batch Prediction":
    st.markdown("""
    <div class="page-header">
        <p class="page-header-title">Batch Prediction</p>
        <p class="page-header-sub">Analyze multiple messages at once via CSV upload</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.875rem;color:#4a5068;margin-bottom:12px">Your CSV must have a column named <code>text</code> containing the messages.</p>', unsafe_allow_html=True)

    sample_df = pd.DataFrame({'text': [
        'Congratulations! You won a FREE prize. Click here now!',
        'Hey are you free for lunch tomorrow?',
        'URGENT: Your account will be suspended. Call now.',
        'Can you send me the notes from class today?'
    ]})
    st.download_button("Download sample CSV", sample_df.to_csv(index=False), "sample_messages.csv", "text/csv")

    uploaded_csv = st.file_uploader("", type=['csv'], label_visibility="collapsed")
    if uploaded_csv:
        try:
            batch_df = pd.read_csv(uploaded_csv)
            if 'text' not in batch_df.columns:
                st.error("CSV must have a column named 'text'")
            elif batch_df.empty:
                st.error("CSV has no rows to analyze.")
            else:
                st.success(f"Loaded {len(batch_df)} messages")
                batch_model_choice = st.selectbox("Select Model", [
                    "SVM + TF-IDF",
                    "DistilBERT Transformer"
                ], key="batch_model")
                if st.button("Analyze all", use_container_width=True):
                    progress = st.progress(0)
                    results = []
                    for i, msg in enumerate(batch_df['text'].fillna("")):
                        if batch_model_choice == "DistilBERT Transformer":
                            prediction, conf, _, keywords = run_bert_prediction(str(msg))
                        else:
                            prediction, conf, _, keywords = run_prediction(str(msg), model)
                        result_label = 'SPAM' if prediction == 1 else 'SAFE'
                        confidence_label = f"{conf:.1%}"
                        results.append({
                            'Message': str(msg)[:80] + ('...' if len(str(msg)) > 80 else ''),
                            'Result': result_label,
                            'Confidence': confidence_label,
                            'Trigger Words': len(keywords),
                            'Model': batch_model_choice
                        })
                        st.session_state.history.append({
                            'message': str(msg)[:55] + ('...' if len(str(msg)) > 55 else ''),
                            'result': result_label,
                            'confidence': confidence_label,
                            'keywords': len(keywords),
                            'source': 'Batch',
                            'model': batch_model_choice
                        })
                        progress.progress((i + 1) / len(batch_df))
                    results_df = pd.DataFrame(results)
                    spam_total = sum(1 for r in results if r['Result'] == 'SPAM')
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("Analyzed", len(results))
                    with c2: st.metric("Spam", spam_total)
                    with c3: st.metric("Safe", len(results) - spam_total)
                    with c4: st.metric("Spam Rate", f"{(spam_total/len(results)*100):.1f}%")
                    st.dataframe(results_df, use_container_width=True)
                    st.download_button("Download results", results_df.to_csv(index=False), "results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════
# PAGE 5 — IMAGE SCANNER
# ══════════════════════════════════════════════════════════
elif page == "Image Scanner":
    st.markdown("""
    <div class="page-header">
        <p class="page-header-title">Image Scanner</p>
        <p class="page-header-sub">Upload an image containing text and scan it for spam</p>
    </div>""", unsafe_allow_html=True)

    uploaded_image = st.file_uploader("", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption=uploaded_image.name, use_column_width=True)
        image_model_choice = st.selectbox("Select Model", [
        "SVM + TF-IDF",
        "DistilBERT Transformer"
    ], key="image_model")

        if st.button("Extract text and scan", use_container_width=True):
            with st.spinner("Reading text from image..."):
                reader = easyocr.Reader(['en'], gpu=False)
                img_array = np.array(image)
                results = reader.readtext(img_array, detail=0)
                extracted_text = ' '.join(results)
            if not extracted_text.strip():
                st.error("Could not extract any text from this image.")
            else:
                st.success(f"Extracted {len(extracted_text)} characters")
                st.text_area("Extracted text", value=extracted_text, height=120, disabled=True)
                if image_model_choice == "DistilBERT Transformer":
                    prediction, confidence, cleaned, found_keywords = run_bert_prediction(extracted_text)
                else:
                    prediction, confidence, cleaned, found_keywords = run_prediction(extracted_text, model)
                render_result(prediction, confidence, found_keywords, source_label=uploaded_image.name)
                st.session_state.history.append({
                    'message': extracted_text[:55] + ('...' if len(extracted_text) > 55 else ''),
                    'result': 'SPAM' if prediction == 1 else 'SAFE',
                    'confidence': f"{confidence:.1%}",
                    'keywords': len(found_keywords),
                    'source': f"Image ({uploaded_image.name})",
                    'model': image_model_choice
                })

# ══════════════════════════════════════════════════════════
# PAGE 6 — DATA EXPLORER
# ══════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown("""
    <div class="page-header">
        <p class="page-header-title">Data Explorer</p>
        <p class="page-header-sub">Browse the dataset used to train the model</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total", f"{len(df):,}")
    with c2: st.metric("Spam", f"{sum(df['label']=='spam'):,}")
    with c3: st.metric("Ham", f"{sum(df['label']=='ham'):,}")
    with c4: st.metric("Spam %", f"{sum(df['label']=='spam')/len(df)*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1: filter_option = st.selectbox("Filter", ["All", "Spam only", "Ham only"])
    with col2: search_term = st.text_input("Search", placeholder="e.g. free, win...")

    if filter_option == "Spam only":
        display_df = df[df['label'] == 'spam'][['label', 'text']].reset_index(drop=True)
    elif filter_option == "Ham only":
        display_df = df[df['label'] == 'ham'][['label', 'text']].reset_index(drop=True)
    else:
        display_df = df[['label', 'text']].reset_index(drop=True)

    if search_term:
        display_df = display_df[display_df['text'].str.contains(search_term, case=False, na=False)]
        st.caption(f"{len(display_df)} results for '{search_term}'")

    st.dataframe(display_df.head(50), use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="sec-label">Text length statistics</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Spam")
        st.write(df[df['label']=='spam']['text'].str.len().describe().round(1))
    with col2:
        st.caption("Ham")
        st.write(df[df['label']=='ham']['text'].str.len().describe().round(1))

# ══════════════════════════════════════════════════════════
# PAGE 7 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════
elif page == "Visualizations":
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from sklearn.feature_extraction.text import CountVectorizer

    st.markdown("""
    <div class="page-header">
        <p class="page-header-title">Visualizations</p>
        <p class="page-header-sub">Visual insights from the dataset and model performance</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribution", "Word Cloud", "Top Words", "Confusion Matrix", "Model Comparison"
    ])

    with tab1:
        counts = df['label'].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='#12121f')
            ax.set_facecolor('#12121f')
            ax.bar(counts.index, counts.values, color=['#7c3aed', '#2d1060'], width=0.4, edgecolor='none')
            ax.set_title('Message Count by Class', color='#e2e4ee', fontsize=11, fontweight='600', pad=12)
            ax.set_ylabel('Count', color='#4a5068', fontsize=9)
            ax.tick_params(colors='#4a5068', labelsize=9)
            for spine in ax.spines.values(): spine.set_color('#1e1e35')
            for i, v in enumerate(counts.values):
                ax.text(i, v + 30, str(v), ha='center', fontsize=9, color='#e2e4ee', fontweight='600')
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='#12121f')
            ax.set_facecolor('#12121f')
            wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index,
                autopct='%1.1f%%', colors=['#7c3aed', '#2d1060'], startangle=90)
            for t in texts: t.set_color('#4a5068'); t.set_fontsize(9)
            for t in autotexts: t.set_color('#e2e4ee'); t.set_fontsize(9); t.set_fontweight('600')
            ax.set_title('Spam vs Ham Proportion', color='#e2e4ee', fontsize=11, fontweight='600', pad=12)
            st.pyplot(fig)

    with tab2:
        wc_option = st.radio("Type", ["Spam", "Ham"], horizontal=True)
        label_filter = 'spam' if wc_option == 'Spam' else 'ham'
        text_data = ' '.join(df[df['label'] == label_filter]['clean_text'].dropna())
        wc_color = 'RdPu' if label_filter == 'spam' else 'Purples'
        wc = WordCloud(width=900, height=350, background_color='#0d0d18',
                       colormap=wc_color, max_words=100).generate(text_data)
        fig, ax = plt.subplots(figsize=(11, 4), facecolor='#0d0d18')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    with tab3:
        top_option = st.radio("Type", ["Spam", "Ham"], horizontal=True, key="top20")
        top_label = 'spam' if top_option == 'Spam' else 'ham'
        top_text = df[df['label'] == top_label]['clean_text'].dropna()
        cv = CountVectorizer(max_features=20)
        word_freq = pd.DataFrame({
            'word': cv.fit(top_text).get_feature_names_out(),
            'count': cv.fit_transform(top_text).toarray().sum(axis=0)
        }).sort_values('count', ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#12121f')
        ax.set_facecolor('#12121f')
        ax.barh(word_freq['word'], word_freq['count'],
                color='#7c3aed' if top_label == 'spam' else '#5b21b6')
        ax.set_title(f'Top 20 Words — {top_option}', color='#e2e4ee', fontsize=11, fontweight='600')
        ax.set_xlabel('Frequency', color='#4a5068', fontsize=9)
        ax.tick_params(colors='#4a5068', labelsize=9)
        for spine in ax.spines.values(): spine.set_color('#1e1e35')
        st.pyplot(fig)

    with tab4:
        try: st.image('data/confusion_matrix.png', use_container_width=True)
        except: st.info("confusion_matrix.png not found in data/ folder.")
        try: st.image('data/bert_confusion_matrix.png', width=600)
        except: st.info("bert_confusion_matrix.png not found in data/ folder.")


    with tab5:
        try: st.image('data/model_comparison_chart.png', use_container_width=True)
        except: st.info("model_comparison_chart.png not found in data/ folder.")

# ══════════════════════════════════════════════════════════
# PAGE 8 — MODEL INFO
# ══════════════════════════════════════════════════════════
elif page == "Model Info":
    st.markdown("""
    <div class="page-header">
        <p class="page-header-title">Model Info</p>
        <p class="page-header-sub">Technical details about the NLP pipeline and models</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec-label">Best model — SVM with TF-IDF</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Accuracy", "98.3%")
    with c2: st.metric("Precision", "99.2%")
    with c3: st.metric("Recall", "87.9%")
    with c4: st.metric("F1-Score", "93.2%")

    st.markdown('<p class="sec-label">Model comparison</p>', unsafe_allow_html=True)
    try:
        results_df = pd.read_csv('data/model_comparison.csv', index_col=0)
        models = results_df.index.tolist()
        accuracies = results_df['Accuracy'].tolist()
        precisions = results_df['Precision'].tolist()
        recalls    = results_df['Recall'].tolist()
        f1s        = results_df['F1-Score'].tolist()
    except:
        models     = ['Naive Bayes (TF-IDF)', 'SVM (TF-IDF)', 'SVM (Word2Vec)', 'BERT Transformer']
        accuracies = [0.9668, 0.9830, 0.9453, 0.9964]
        precisions = [0.9912, 0.9924, 0.8729, 1.0000]
        recalls    = [0.7584, 0.8792, 0.6913, 0.9732]
        f1s        = [0.8593, 0.9324, 0.7715, 0.9864]

    best_idx = f1s.index(max(f1s))
    display_models = [m + "  — Best" if i == best_idx else m for i, m in enumerate(models)]

    table_df = pd.DataFrame({
        'Model': display_models,
        'Accuracy': accuracies, 'Precision': precisions,
        'Recall': recalls, 'F1-Score': f1s
    })

    def highlight_best(row):
        if '— Best' in str(row['Model']):
            return ['background-color:#052e16;color:#86efac;font-weight:700'] * len(row)
        return ['background-color:#12121f;color:#8890a8'] * len(row)

    styled = (table_df.style
        .apply(highlight_best, axis=1)
        .format({'Accuracy':'{:.4f}','Precision':'{:.4f}','Recall':'{:.4f}','F1-Score':'{:.4f}'})
        .set_table_styles([
            {'selector':'thead th','props':[('background-color','#0d0d18'),('color','#3a3f5c'),
                ('font-size','0.72rem'),('text-transform','uppercase'),('letter-spacing','0.08em'),
                ('padding','14px 16px'),('border-bottom','2px solid #1e1e35'),('font-weight','700')]},
            {'selector':'td','props':[('padding','14px 16px'),('border-bottom','1px solid #1a1a2e'),
                ('font-size','0.9rem')]},
            {'selector':'table','props':[('border-collapse','collapse'),('width','100%')]},
        ]).hide(axis='index'))

    st.dataframe(styled, use_container_width=True, hide_index=True, height=168)
    st.markdown("---")

    st.markdown('<p class="sec-label" style="margin-top:20px">Advanced NLP — BERT Transformer</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="info-card" style="border-left:4px solid #7c3aed">
    <div style='display:flex;align-items:flex-start;gap:16px'>
        <div style='flex:1'>
            <div style='font-size:0.7rem;font-weight:700;color:#7c3aed;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;font-family:"Fira Code",monospace'>BERT · Transformers · HuggingFace</div>
            <div style='font-size:0.875rem;color:#8890a8;line-height:1.7'>
            BERT Transformer is a deep learning model that understands the full context of a message by reading words in both directions at once. It was pre-trained on large text corpora and fine-tuned specifically for SMS spam detection.
            Unlike SVM which looks at word frequencies, BERT understands the <em>meaning</em> behind words based on surrounding context — making it a more powerful and flexible language model.
            </div>
        </div>
    </div>
    <div style='display:flex;gap:24px;margin-top:16px;flex-wrap:wrap'>
        <div style='background:#1a0a3d;border:1px solid #3d1e7a;border-radius:8px;padding:10px 16px;min-width:120px'>
            <div style='font-size:0.65rem;color:#4a5068;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px'>Model</div>
            <div style='font-size:0.8rem;color:#a78bfa;font-family:"Fira Code",monospace;font-weight:600'>DistilBERT</div>
        </div>
        <div style='background:#1a0a3d;border:1px solid #3d1e7a;border-radius:8px;padding:10px 16px;min-width:120px'>
            <div style='font-size:0.65rem;color:#4a5068;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px'>Library</div>
            <div style='font-size:0.8rem;color:#a78bfa;font-family:"Fira Code",monospace;font-weight:600'>HuggingFace</div>
        </div>
        <div style='background:#1a0a3d;border:1px solid #3d1e7a;border-radius:8px;padding:10px 16px;min-width:120px'>
            <div style='font-size:0.65rem;color:#4a5068;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px'>Max Input</div>
            <div style='font-size:0.8rem;color:#a78bfa;font-family:"Fira Code",monospace;font-weight:600'>512 tokens</div>
        </div>
        <div style='background:#1a0a3d;border:1px solid #3d1e7a;border-radius:8px;padding:10px 16px;min-width:120px'>
            <div style='font-size:0.65rem;color:#4a5068;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px'>Type</div>
            <div style='font-size:0.8rem;color:#a78bfa;font-family:"Fira Code",monospace;font-weight:600'>Transformer</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    try:
        results_df = pd.read_csv('data/bert_model.csv', index_col=0)
        models = results_df.index.tolist()
        accuracies = results_df['Accuracy'].tolist()
        precisions = results_df['Precision'].tolist()
        recalls    = results_df['Recall'].tolist()
        f1s        = results_df['F1-Score'].tolist()
    except:
        models     = ['DistilBERT']
        accuracies = [ 0.9964]
        precisions = [ 1.0000]
        recalls    = [0.9732]
        f1s        = [0.9864]
    
    best_idx = f1s.index(max(f1s))
    display_models = [m for i, m in enumerate(models)]

    table_df = pd.DataFrame({
        'Model': display_models,
        'Accuracy': accuracies, 'Precision': precisions,
        'Recall': recalls, 'F1-Score': f1s
    })

    def highlight_best(row):
        if 'BERT' in str(row['Model']):
            return ['background-color:#12121f;color:#8890a8'] * len(row)

    styled = (table_df.style
        .apply(highlight_best, axis=1)
        .format({'Accuracy':'{:.4f}','Precision':'{:.4f}','Recall':'{:.4f}','F1-Score':'{:.4f}'})
        .set_table_styles([
            {'selector':'thead th','props':[('background-color','#0d0d18'),('color','#3a3f5c'),
                ('font-size','0.72rem'),('text-transform','uppercase'),('letter-spacing','0.08em'),
                ('padding','14px 16px'),('border-bottom','2px solid #1e1e35'),('font-weight','700')]},
            {'selector':'td','props':[('padding','14px 16px'),('border-bottom','1px solid #1a1a2e'),
                ('font-size','0.9rem')]},
            {'selector':'table','props':[('border-collapse','collapse'),('width','100%')]},
        ]).hide(axis='index'))

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=68)


    st.markdown("---")
    st.markdown('<p class="sec-label">NLP Pipeline</p>', unsafe_allow_html=True)
    st.markdown("""
    | Step | Method | Description |
    |------|--------|-------------|
    | 1 | Lowercasing | Convert all text to lowercase |
    | 2 | URL Removal | Remove http/www links |
    | 3 | Cleaning | Remove numbers and special characters |
    | 4 | Tokenization | Split into individual words |
    | 5 | Stopword Removal | Remove common words like the, is, and |
    | 6 | Lemmatization | Reduce words to their base form |
    """)

    st.markdown("---")
    st.markdown('<p class="sec-label">Models</p>', unsafe_allow_html=True)
    with st.expander("Naive Bayes (TF-IDF)"):
        st.write("A probabilistic classifier based on Bayes theorem. Fast, simple, and historically the most popular algorithm for spam filtering. Accuracy: 96.7% | F1: 85.9%")
    with st.expander("SVM with TF-IDF — Deployed model"):
        st.write("Finds the optimal boundary separating spam from ham. Handles high-dimensional TF-IDF feature spaces very well and consistently outperforms Naive Bayes. Accuracy: 98.3% | F1: 93.2%")
    with st.expander("SVM with Word2Vec"):
        st.write("Uses averaged Word2Vec embeddings. Captures semantic meaning but loses keyword signals important for spam detection. Accuracy: 94.6% | F1: 77.4%")
    with st.expander("BERT Transformer — Advanced NLP"):
        st.markdown("""
    <div style='font-size:0.875rem;color:#8890a8;line-height:1.75'>
    <strong style='color:#a78bfa'>BERT (Bidirectional Encoder Representations from Transformers)</strong>
    is a state-of-the-art deep learning model developed by Google in 2018. Unlike traditional
    models that read text from left to right, BERT reads the entire sentence at once in both
    directions, giving it a much deeper understanding of context and word meaning.
    <br><br>
    We used a lightweight and faster version called <strong style='color:#a78bfa'>DistilBERT Transformer</strong>
    from HuggingFace, which is pre-trained on a massive English text corpus and then
    fine-tuned specifically on SMS spam data — the same type of data as our dataset.
    <br><br>
    <strong style='color:#c5c8dc'>Why BERT is different from SVM and Naive Bayes:</strong><br>
    Traditional models like SVM and Naive Bayes rely on word frequency and statistical patterns.
    BERT actually understands the meaning and context of each word in a sentence. For example,
    the word "free" in "feel free to call me" is treated differently from "free" in "win a free prize".
    <br><br>
    <strong style='color:#c5c8dc'>Library used:</strong> HuggingFace Transformers<br>
    <strong style='color:#c5c8dc'>Model:</strong> DistilBERT<br>
    <strong style='color:#c5c8dc'>Input limit:</strong> 512 tokens per message<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="sec-label">Feature extraction</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("TF-IDF"):
            st.write("Weighs words by importance relative to the full dataset. Words like FREE and WIN score high in spam. Vocabulary: 5,000 features.")
    with col2:
        with st.expander("Word2Vec"):
            st.write("Neural embeddings capturing semantic meaning. Each message is the average of its 100-dimension word vectors. Vocabulary: 7,807 words.")

# ══════════════════════════════════════════════════════════
# PAGE 9 — HISTORY
# ══════════════════════════════════════════════════════════
elif page == "History":
    st.markdown("""
    <div class="page-header">
        <p class="page-header-title">Session History</p>
        <p class="page-header-sub">All messages analyzed in this session</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div style="background:#0d0d18;border:1.5px dashed #1e1e35;border-radius:12px;padding:60px;text-align:center">
            <p style="font-size:0.875rem;color:#2d3147;margin:0">No analysis history yet. Start by analyzing a message in Text Analyzer.</p>
        </div>""", unsafe_allow_html=True)
    else:
        total = len(st.session_state.history)
        spam_count = sum(1 for h in st.session_state.history if h['result'] == 'SPAM')
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total", total)
        with c2: st.metric("Spam", spam_count)
        with c3: st.metric("Safe", total - spam_count)
        with c4: st.metric("Spam Rate", f"{(spam_count/total*100):.1f}%" if total > 0 else "—")

        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1: filter_result = st.selectbox("Filter by result", ["All", "Spam only", "Safe only"])
        with col2: filter_source = st.selectbox("Filter by source", ["All", "Text", "File", "Image", "Batch"])

        filtered = st.session_state.history.copy()
        if filter_result == "Spam only": filtered = [h for h in filtered if h['result'] == 'SPAM']
        elif filter_result == "Safe only": filtered = [h for h in filtered if h['result'] == 'SAFE']
        if filter_source != "All": filtered = [h for h in filtered if str(h.get('source', '')).startswith(filter_source)]

        st.markdown(f'<p class="history-count">Showing {len(filtered)} of {total} records</p>', unsafe_allow_html=True)

        st.markdown("---")
        for i, item in enumerate(reversed(filtered)):
            css = "hist-item" if item['result'] == 'SPAM' else "hist-item ham"
            st.markdown(f"""
            <div class="{css}">
                <div class="hist-msg"><strong>{item['result']}</strong> — {item['message']}</div>
                <div class="hist-meta">Confidence: {item['confidence']} &nbsp;·&nbsp; Trigger words: {item['keywords']} &nbsp;·&nbsp; Source: {item['source']} &nbsp;·&nbsp; Model: {item.get('model', 'SVM + TF-IDF')} &nbsp;·&nbsp; #{total - i}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        export_df = pd.DataFrame(st.session_state.history)
        st.download_button("DOWNLOAD HISTORY AS CSV", export_df.to_csv(index=False),
                           "analysis_history.csv", "text/csv", use_container_width=True)
        if st.button("CLEAR ALL HISTORY", use_container_width=True):
            st.session_state.history = []
            st.rerun()
