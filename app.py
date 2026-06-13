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

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background: #000000;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0a0a0a;
        border-right: 1px solid #2a2a2a;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 14px;
        padding: 20px;
    }

    [data-testid="metric-container"] label {
        color: #888888 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Hero banner */
    .hero-banner {
        background: #111111;
        border: 1px solid #333333;
        border-top: 3px solid #ffffff;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin-bottom: 24px;
    }
    .hero-title {
    font-size: 3.5rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -1px;
    line-height: 1.3;
    font-family: 'IBM Plex Mono', monospace;
    }
    .hero-subtitle {
        color: #888888;
        font-size: 1rem;
        margin: 0;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-size: 0.8rem;
    }

    /* Result boxes */
    .spam-box {
        background: #1a0808;
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 0 24px rgba(239,68,68,0.15);
    }
    .ham-box {
        background: #081a0e;
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 0 24px rgba(34,197,94,0.15);
    }
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -1px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .spam-box .result-label { color: #ef4444; }
    .ham-box .result-label  { color: #22c55e; }
    .result-sub {
        font-size: 0.85rem;
        color: #aaaaaa;
        margin: 0;
        letter-spacing: 0.5px;
    }

    /* Spam meter */
    .meter-wrap {
        background: #111111;
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 12px 0;
    }
    .meter-label {
        font-size: 0.75rem;
        color: #888888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .meter-bar-bg {
        background: #222222;
        border-radius: 99px;
        height: 12px;
        overflow: hidden;
    }
    .meter-bar-fill-spam {
        height: 100%;
        background: linear-gradient(90deg, #f59e0b, #ef4444);
        border-radius: 99px;
        transition: width 0.8s ease;
    }
    .meter-bar-fill-ham {
        height: 100%;
        background: linear-gradient(90deg, #34d399, #22c55e);
        border-radius: 99px;
        transition: width 0.8s ease;
    }
    .meter-value-spam {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 10px;
        color: #ef4444;
        font-family: 'IBM Plex Mono', monospace;
    }
    .meter-value-ham {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 10px;
        color: #22c55e;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Keyword badges */
    .keyword-badge {
        display: inline-block;
        background: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.4);
        color: #fca5a5;
        border-radius: 99px;
        padding: 4px 14px;
        margin: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.5px;
    }

    /* History items */
    .history-item {
        background: #111111;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 8px;
        border-left: 3px solid #ef4444;
    }
    .history-item.ham {
        border-left-color: #22c55e;
    }
    .history-msg {
        font-size: 0.9rem;
        color: #dddddd;
    }
    .history-meta {
        font-size: 0.75rem;
        color: #666666;
        margin-top: 4px;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Feature cards */
    .feature-card {
        background: #111111;
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        height: 100%;
    }
    .feature-icon {
        font-size: 1.2rem;
        margin-bottom: 10px;
        color: #888888;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.7rem;
    }
    .feature-title {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.82rem;
        color: #777777;
        line-height: 1.5;
    }

    /* Upload zone */
    .upload-zone {
        background: #0d0d0d;
        border: 2px dashed #333333;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 24px 0 16px 0;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid #222222;
        padding-bottom: 10px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #111111;
        border-radius: 12px;
        padding: 6px;
        gap: 6px;
        border: 1px solid #2a2a2a;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #888888 !important;
        font-weight: 600;
        font-size: 0.8rem;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 8px 20px !important;
        white-space: nowrap;
    }
    .stTabs [aria-selected="true"] {
        background: #2a2a2a !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
    }
    /* Tab content panel */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }

    /* Buttons */
    .stButton > button {
        background: #222222 !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-size: 0.8rem !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        background: #333333 !important;
        border-color: #666666 !important;
    }

    /* Inputs and textareas */
    .stTextArea textarea {
        background: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 12px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.88rem !important;
    }
    .stTextInput input {
        background: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 12px !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 12px !important;
    }
    .stSelectbox svg { color: #888888 !important; fill: #888888 !important; }

    /* Dropdown options */
    [data-baseweb="popover"] { background: #1a1a1a !important; border: 1px solid #333 !important; }
    [data-baseweb="menu"] { background: #1a1a1a !important; }
    [role="option"] { background: #1a1a1a !important; color: #ffffff !important; }
    [role="option"]:hover { background: #2a2a2a !important; }

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #777777 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
    }

    /* Info / success / warning / error boxes */
    .stAlert {
        background: #111111 !important;
        border: 1px solid #333333 !important;
        border-radius: 12px !important;
        color: #cccccc !important;
    }
    .stAlert p { color: #cccccc !important; }

    /* Expander */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        background: #111111 !important;
        color: #cccccc !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 12px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stExpander"] { border: 1px solid #2a2a2a !important; border-radius: 12px !important; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #2a2a2a !important; border-radius: 12px !important; }
    .stDataFrame th { background: #1a1a1a !important; color: #aaaaaa !important; }
    .stDataFrame td { background: #111111 !important; color: #dddddd !important; }

    /* Code blocks */
    .stCode, [data-testid="stCode"] {
        background: #111111 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 10px !important;
    }
    .stCode code, [data-testid="stCode"] code { color: #dddddd !important; }

    /* Radio buttons */
    .stRadio label { color: #cccccc !important; }
    .stRadio [data-testid="stWidgetLabel"] p { color: #cccccc !important; }

    /* Checkbox */
    .stCheckbox label { color: #cccccc !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #111111 !important;
        border: 1px dashed #444444 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span { color: #888888 !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] { color: #888888 !important; }

    /* Download button */
    [data-testid="stDownloadButton"] > button {
        background: #222222 !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
        border-radius: 12px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: #333333 !important;
        border-color: #666666 !important;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: #ffffff !important; }

    /* General text — force all white/light */
    p, li, span, div, label, .stMarkdown, .stMarkdown p {
        color: #cccccc;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    strong { color: #ffffff !important; }

    /* Horizontal rule */
    hr { border-color: #2a2a2a !important; }

    /* Tables (markdown only - not custom HTML tables) */
    .stMarkdown table { border-collapse: collapse; width: 100%; }
    .stMarkdown th { background: #1a1a1a !important; color: #888888 !important; padding: 10px 14px; border: 1px solid #2a2a2a; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .stMarkdown td { background: #111111 !important; color: #dddddd !important; padding: 10px 14px; border: 1px solid #1e1e1e; }
    .stMarkdown tr:hover td { background: #181818 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #111111; }
    ::-webkit-scrollbar-thumb { background: #333333; border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: #555555; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar nav text */
    [data-testid="stSidebar"] .stRadio label {
        color: #aaaaaa !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Sidebar metrics */
    [data-testid="stSidebar"] [data-testid="metric-container"] {
        background: #0a0a0a !important;
        border-color: #222222 !important;
    }

    /* Progress bar */
    .stProgress > div > div { background: #ffffff !important; }
    .stProgress > div { background: #222222 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Load Model & Vectorizer ───────────────────────────────
@st.cache_resource
def load_model():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ── Load Dataset ──────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data/spam_cleaned.csv')

df = load_data()

# ── Preprocessing ─────────────────────────────────────────
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

def get_confidence(decision_value):
    return float(1 / (1 + np.exp(-abs(decision_value))))

def run_prediction(text, selected_model):
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = selected_model.predict(vectorized)[0]
    decision = selected_model.decision_function(vectorized)[0]
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
        text = ""
        st.error(f"Could not read file: {e}")
    return text.strip()

def render_result_box(prediction, confidence, found_keywords, source_label=""):
    if prediction == 1:
        st.markdown(f"""
        <div class="spam-box">
            <p class="result-label">SPAM DETECTED</p>
            <p class="result-sub">{source_label}Confidence: {confidence:.1%} &nbsp;&middot;&nbsp; {len(found_keywords)} spam indicators found</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ham-box">
            <p class="result-label">SAFE MESSAGE</p>
            <p class="result-sub">{source_label}Confidence: {confidence:.1%} &nbsp;&middot;&nbsp; No major spam indicators</p>
        </div>""", unsafe_allow_html=True)

    bar_class = "meter-bar-fill-spam" if prediction == 1 else "meter-bar-fill-ham"
    val_class = "meter-value-spam" if prediction == 1 else "meter-value-ham"
    pct = int(confidence * 100)
    st.markdown(f"""
    <div class="meter-wrap">
        <div class="meter-label">Spam Probability</div>
        <div class="meter-bar-bg">
            <div class="{bar_class}" style="width:{pct}%"></div>
        </div>
        <div class="{val_class}">{pct}%</div>
    </div>""", unsafe_allow_html=True)

    if found_keywords:
        st.markdown("**Spam Trigger Words Detected:**")
        badges = "".join([f'<span class="keyword-badge">{w}</span>' for w in found_keywords])
        st.markdown(badges, unsafe_allow_html=True)
    else:
        st.info("No common spam trigger words found.")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 24px 0;'>
        <div style='font-size:0.7rem; font-weight:700; color:#555555; letter-spacing:4px; font-family:"IBM Plex Mono",monospace; text-transform:uppercase; margin-bottom:6px'>SYSTEM</div>
        <div style='font-size:1.5rem; font-weight:700; color:#ffffff; letter-spacing:-1px; font-family:"IBM Plex Mono",monospace'>SpamShield</div>
        <div style='font-size:0.65rem; color:#555555; margin-top:4px; font-family:"IBM Plex Mono",monospace; text-transform:uppercase; letter-spacing:2px'>SVM + NLP Engine</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigate", [
        "Home",
        "Text Analyzer",
        "File Scanner",
        "Batch Prediction",
        "Image Scanner",
        "Data Explorer",
        "Visualizations",
        "Model Info",
        "History"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Session Stats")
    total = len(st.session_state.history)
    spam_count = sum(1 for h in st.session_state.history if h['result'] == 'SPAM')
    ham_count = total - spam_count

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Analyzed", total)
        st.metric("Spam", spam_count)
    with col2:
        st.metric("Safe", ham_count)
        rate = f"{(spam_count/total*100):.0f}%" if total > 0 else "—"
        st.metric("Spam Rate", rate)

    if total > 0:
        if st.button("Clear Session", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">SpamShield AI</p>
        <p class="hero-subtitle">Intelligent spam detection powered by NLP and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Data", f"{len(df):,}")
    with col2:
        st.metric("Spam Samples", f"{df['label'].value_counts().get('spam', 0):,}")
    with col3:
        st.metric("Ham Samples", f"{df['label'].value_counts().get('ham', 0):,}")
    with col4:
        st.metric("Best F1-Score", "93.2%")

    st.markdown("---")
    st.markdown("### What You Can Do")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    features = [
        ("[ TEXT ]", "Text Analyzer", "Type or paste any message and get instant spam detection with confidence score"),
        ("[ FILE ]", "File Scanner", "Upload a .txt, .pdf or .docx file and scan its content for spam"),
        ("[ IMAGE ]", "Image Scanner", "Upload an image containing text and extract it via OCR for spam scanning"),
        ("[ BATCH ]", "Batch Prediction", "Upload a CSV with multiple messages and analyze them all at once"),
        ("[ DATA ]", "Visualizations", "Explore word clouds, model comparisons, and dataset insights"),
        ("[ LOG ]", "History", "Review and export every message analyzed during this session"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4, c5, c6], features):
        with col:
            st.markdown(f"""
            <div class="feature-card" style="margin-bottom:16px">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
            
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### About This Project")
        st.info("""
        This application uses NLP and Machine Learning to automatically detect
        whether a message is spam or legitimate (ham).

        Trained on the SMS Spam Collection dataset with 5,572 messages.
        Three models were compared: Naive Bayes, SVM (TF-IDF), and SVM (Word2Vec).
        The best model — SVM with TF-IDF — was deployed with 98.3% accuracy.
        """)
    with col2:
        st.markdown("### Team Members")
        st.markdown("""
        <div class="feature-card" style="text-align:left">
            <div style="margin-bottom:12px; color:#cccccc"><strong style="color:#ffffff">Member 1</strong> — Data & NLP Pipeline</div>
            <div style="margin-bottom:12px; color:#cccccc"><strong style="color:#ffffff">Member 2</strong> — Model Training & Evaluation</div>
            <div style="margin-bottom:12px; color:#cccccc"><strong style="color:#ffffff">Member 3</strong> — Streamlit Application</div>
            <div style="color:#cccccc"><strong style="color:#ffffff">Member 4</strong> — Visualizations & Documentation</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — TEXT ANALYZER
# ══════════════════════════════════════════════════════════
elif page == "Text Analyzer":
    st.markdown('<div class="hero-banner"><p class="hero-title" style="font-size:2rem">Text Analyzer</p><p class="hero-subtitle">Paste any message to instantly detect spam</p></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        user_input = st.text_area(
            "Enter your message:",
            height=180,
            placeholder="Paste your email or SMS here...\n\nExample: Congratulations! You've won a FREE prize. Click here to claim now!",
            key="text_input"
        )

        char_count = len(user_input)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.caption(f"{char_count} characters")
        with col_b:
            word_count = len(user_input.split()) if user_input else 0
            st.caption(f"{word_count} words")
        with col_c:
            if char_count > 138:
                st.caption("Above avg spam length")
            else:
                st.caption("Normal length")

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            analyze_btn = st.button("Analyze Message", use_container_width=True)
        with col_btn2:
            if st.button("Clear", use_container_width=True):
                st.rerun()

        st.markdown("##### Try these examples:")
        tab1, tab2 = st.tabs(["Spam Example", "Ham Example"])
        with tab1:
            st.code("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question std txt rate")
        with tab2:
            st.code("Hey, are you coming to the meeting tomorrow at 10am? Let me know if you need the slides.")

    with col_right:
        if analyze_btn:
            if not user_input.strip():
                st.warning("Please enter a message first.")
            else:
                with st.spinner("Analyzing..."):
                    prediction, confidence, cleaned, found_keywords = run_prediction(user_input, model)

                render_result_box(prediction, confidence, found_keywords)

                with st.expander("Preprocessing Details"):
                    st.markdown("**After cleaning:**")
                    st.code(cleaned, language=None)

                st.session_state.history.append({
                    'message': user_input[:55] + ('...' if len(user_input) > 55 else ''),
                    'result': 'SPAM' if prediction == 1 else 'HAM',
                    'confidence': f"{confidence:.1%}",
                    'keywords': len(found_keywords),
                    'source': 'Text'
                })
        else:
            st.markdown("""
            <div style="background:#0d0d0d; border:1px dashed #2a2a2a; border-radius:16px; padding:40px; text-align:center; color:#444444">
                <div style="font-size:0.7rem; font-family:'IBM Plex Mono',monospace; text-transform:uppercase; letter-spacing:3px; margin-bottom:12px">[ READY ]</div>
                <div style="font-size:0.9rem; color:#555555">Enter a message and click<br><strong style="color:#aaaaaa">Analyze</strong> to see results here</div>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Recent Analysis History")
        for item in reversed(st.session_state.history[-6:]):
            css = "history-item" if item['result'] == 'SPAM' else "history-item ham"
            st.markdown(f"""
            <div class="{css}">
                <div class="history-msg"><strong>{item['result']}</strong> — {item['message']}</div>
                <div class="history-meta">Confidence: {item['confidence']} &nbsp;&middot;&nbsp; Trigger words: {item['keywords']} &nbsp;&middot;&nbsp; Source: {item['source']}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — FILE SCANNER
# ══════════════════════════════════════════════════════════
elif page == "File Scanner":
    st.markdown('<div class="hero-banner"><p class="hero-title" style="font-size:2rem">File Scanner</p><p class="hero-subtitle">Upload a file and scan its content for spam</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:0.65rem; font-family:'IBM Plex Mono',monospace; text-transform:uppercase; letter-spacing:3px; color:#555555; margin-bottom:10px">Supported Formats</div>
        <div style="color:#aaaaaa; font-size:1rem; font-family:'IBM Plex Mono',monospace">.txt &nbsp;&middot;&nbsp; .pdf &nbsp;&middot;&nbsp; .docx</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload file:",
        type=['txt', 'pdf', 'docx'],
        label_visibility="collapsed"
    )

    if uploaded_file:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            extracted_text = extract_text_from_file(uploaded_file)

        if not extracted_text:
            st.error("Could not extract text from this file. Please try a different file.")
        else:
            st.success(f"Extracted {len(extracted_text):,} characters from {uploaded_file.name}")

            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown("##### Extracted Content Preview:")
                st.text_area("", value=extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""),
                             height=200, disabled=True, label_visibility="collapsed")
                if len(extracted_text) > 1000:
                    st.caption(f"Showing first 1,000 of {len(extracted_text):,} characters")

            with col2:
                st.markdown("##### File Stats:")
                st.metric("Characters", f"{len(extracted_text):,}")
                st.metric("Words", f"{len(extracted_text.split()):,}")
                st.metric("Lines", f"{len(extracted_text.splitlines()):,}")

            st.markdown("---")
            if st.button("Scan This File for Spam", use_container_width=True):
                with st.spinner("Scanning content..."):
                    prediction, confidence, cleaned, found_keywords = run_prediction(extracted_text, model)

                render_result_box(prediction, confidence, found_keywords, source_label=f"File: {uploaded_file.name} · ")

                with st.expander("View Preprocessed Text"):
                    st.code(cleaned[:500] + ("..." if len(cleaned) > 500 else ""), language=None)

                st.session_state.history.append({
                    'message': f"[FILE] {uploaded_file.name}",
                    'result': 'SPAM' if prediction == 1 else 'HAM',
                    'confidence': f"{confidence:.1%}",
                    'keywords': len(found_keywords),
                    'source': 'File'
                })

# ══════════════════════════════════════════════════════════
# PAGE 4 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════
elif page == "Batch Prediction":
    st.markdown('<div class="hero-banner"><p class="hero-title" style="font-size:2rem">Batch Prediction</p><p class="hero-subtitle">Analyze multiple messages at once via CSV upload</p></div>', unsafe_allow_html=True)

    st.info("Your CSV must have a column named 'text' containing the messages to analyze.")

    sample_df = pd.DataFrame({'text': [
        'Congratulations! You won a FREE prize. Click here now!',
        'Hey are you free for lunch tomorrow?',
        'URGENT: Your account will be suspended. Call now to claim your reward.',
        'Can you send me the notes from class today?'
    ]})
    st.download_button("Download Sample CSV Template", sample_df.to_csv(index=False),
                       "sample_messages.csv", "text/csv")

    uploaded_csv = st.file_uploader("Upload your CSV:", type=['csv'])

    if uploaded_csv:
        try:
            batch_df = pd.read_csv(uploaded_csv)
            if 'text' not in batch_df.columns:
                st.error("CSV must have a column named 'text'")
            else:
                st.success(f"Loaded {len(batch_df)} messages")

                if st.button("Analyze All Messages", use_container_width=True):
                    progress = st.progress(0)
                    results = []
                    for i, msg in enumerate(batch_df['text']):
                        pred, conf, _, keywords = run_prediction(str(msg), model)
                        results.append({
                            'Message': str(msg)[:80] + ('...' if len(str(msg)) > 80 else ''),
                            'Prediction': 'SPAM' if pred == 1 else 'HAM',
                            'Confidence': f"{conf:.1%}",
                            'Spam Keywords': len(keywords)
                        })
                        progress.progress((i + 1) / len(batch_df))

                    results_df = pd.DataFrame(results)
                    spam_total = sum(1 for r in results if r['Prediction'] == 'SPAM')

                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Total Analyzed", len(results))
                    with col2: st.metric("Spam Found", spam_total)
                    with col3: st.metric("Ham Found", len(results) - spam_total)
                    with col4:
                        rate = f"{(spam_total/len(results)*100):.1f}%"
                        st.metric("Spam Rate", rate)

                    st.dataframe(results_df, use_container_width=True)
                    st.download_button("Download Results as CSV",
                                       results_df.to_csv(index=False),
                                       "spam_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ══════════════════════════════════════════════════════════
# PAGE 5 — IMAGE SCANNER
# ══════════════════════════════════════════════════════════
elif page == "Image Scanner":
    st.markdown('<div class="hero-banner"><p class="hero-title" style="font-size:2rem">Image Scanner</p><p class="hero-subtitle">Upload an image containing text to scan for spam</p></div>', unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload image:", type=['png', 'jpg', 'jpeg'])

    if uploaded_image:
        from PIL import Image
        import easyocr
        import numpy as np

        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Extract Text and Scan", use_container_width=True):
            with st.spinner("Reading text from image..."):
                reader = easyocr.Reader(['en'], gpu=False)
                img_array = np.array(image)
                results = reader.readtext(img_array, detail=0)
                extracted_text = ' '.join(results)

            if not extracted_text.strip():
                st.error("Could not extract any text from this image.")
            else:
                st.success(f"Extracted {len(extracted_text)} characters")
                st.text_area("Extracted Text:", value=extracted_text, height=150, disabled=True)

                prediction, confidence, cleaned, found_keywords = run_prediction(extracted_text, model)
                render_result_box(prediction, confidence, found_keywords, source_label=f"Image: {uploaded_image.name} · ")

                st.session_state.history.append({
                    'message': f"[IMAGE] {uploaded_image.name}",
                    'result': 'SPAM' if prediction == 1 else 'HAM',
                    'confidence': f"{confidence:.1%}",
                    'keywords': len(found_keywords),
                    'source': 'Image'
                })

# ══════════════════════════════════════════════════════════
# PAGE 6 — DATA EXPLORER
# ══════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown('<div class="hero-banner"><p class="hero-title" style="font-size:2rem">Data Explorer</p><p class="hero-subtitle">Explore the training dataset</p></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Samples", f"{len(df):,}")
    with col2: st.metric("Spam", f"{sum(df['label']=='spam'):,}")
    with col3: st.metric("Ham", f"{sum(df['label']=='ham'):,}")
    with col4:
        spam_pct = sum(df['label']=='spam') / len(df) * 100
        st.metric("Spam %", f"{spam_pct:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        filter_option = st.selectbox("Filter:", ["All", "Spam only", "Ham only"])
    with col2:
        search_term = st.text_input("Search:", placeholder="e.g. free, win...")

    if filter_option == "Spam only":
        display_df = df[df['label'] == 'spam'][['label', 'text']].reset_index(drop=True)
    elif filter_option == "Ham only":
        display_df = df[df['label'] == 'ham'][['label', 'text']].reset_index(drop=True)
    else:
        display_df = df[['label', 'text']].reset_index(drop=True)

    if search_term:
        display_df = display_df[display_df['text'].str.contains(search_term, case=False, na=False)]
        st.caption(f"Found {len(display_df)} messages containing '{search_term}'")

    st.dataframe(display_df.head(50), use_container_width=True)

    st.markdown("---")
    st.markdown("### Text Length Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Spam messages:**")
        st.write(df[df['label']=='spam']['text'].str.len().describe().round(1))
    with col2:
        st.markdown("**Ham messages:**")
        st.write(df[df['label']=='ham']['text'].str.len().describe().round(1))

# ══════════════════════════════════════════════════════════
# PAGE 7 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════
elif page == "Visualizations":
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from sklearn.feature_extraction.text import CountVectorizer

    st.markdown('<div class="hero-banner"><p class="hero-title" style="font-size:2rem">Visualizations</p><p class="hero-subtitle">Visual insights from data and model performance</p></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribution", "Word Cloud", "Top Words", "Confusion Matrix", "Model Comparison"
    ])

    with tab1:
        st.markdown("### Spam vs Ham Distribution")
        col1, col2 = st.columns(2)
        counts = df['label'].value_counts()
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='#111111')
            ax.set_facecolor('#111111')
            ax.bar(counts.index, counts.values, color=['#ffffff', '#555555'], width=0.5, edgecolor='none')
            ax.set_title('Message Count by Class', color='white', fontsize=12, fontfamily='monospace')
            ax.set_ylabel('Count', color='#888888')
            ax.tick_params(colors='#888888')
            for spine in ax.spines.values(): spine.set_color('#2a2a2a')
            for i, v in enumerate(counts.values):
                ax.text(i, v + 30, str(v), ha='center', fontweight='bold', color='white')
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='#111111')
            ax.set_facecolor('#111111')
            wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index,
                autopct='%1.1f%%', colors=['#ffffff', '#444444'], startangle=90)
            for t in texts: t.set_color('#aaaaaa')
            for t in autotexts: t.set_color('black'); t.set_fontweight('bold')
            ax.set_title('Proportion of Spam vs Ham', color='white', fontsize=12, fontfamily='monospace')
            st.pyplot(fig)

    with tab2:
        st.markdown("### Word Cloud")
        wc_option = st.radio("Message type:", ["Spam", "Ham"], horizontal=True)
        label_filter = 'spam' if wc_option == 'Spam' else 'ham'
        text_data = ' '.join(df[df['label'] == label_filter]['clean_text'].dropna())
        wc = WordCloud(width=900, height=400, background_color='#111111',
                       colormap='gray',
                       max_words=120).generate(text_data)
        fig, ax = plt.subplots(figsize=(11, 4), facecolor='#111111')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    with tab3:
        st.markdown("### Top 20 Most Frequent Words")
        top_option = st.radio("Select:", ["Spam", "Ham"], horizontal=True, key="top20")
        top_label = 'spam' if top_option == 'Spam' else 'ham'
        top_text = df[df['label'] == top_label]['clean_text'].dropna()
        cv = CountVectorizer(max_features=20)
        word_freq = pd.DataFrame({
            'word': cv.fit(top_text).get_feature_names_out(),
            'count': cv.fit_transform(top_text).toarray().sum(axis=0)
        }).sort_values('count', ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#111111')
        ax.set_facecolor('#111111')
        bar_color = '#ffffff' if top_label == 'spam' else '#888888'
        ax.barh(word_freq['word'], word_freq['count'], color=bar_color)
        ax.set_title(f'Top 20 Words in {top_option} Messages', color='white', fontweight='bold', fontfamily='monospace')
        ax.set_xlabel('Frequency', color='#888888')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values(): spine.set_color('#2a2a2a')
        st.pyplot(fig)

    with tab4:
        st.markdown("### Confusion Matrix")
        try:
            st.image('data/confusion_matrix.png', use_column_width=True)
        except:
            st.info("confusion_matrix.png not found in data/ folder.")

    with tab5:
        st.markdown("### Model Performance Comparison")
        try:
            st.image('data/model_comparison_chart.png', use_column_width=True)
        except:
            st.info("model_comparison_chart.png not found in data/ folder.")

# ══════════════════════════════════════════════════════════
# PAGE 8 — MODEL INFO
# ══════════════════════════════════════════════════════════
elif page == "Model Info":
    st.markdown('<div class="hero-banner"><p class="hero-title" style="font-size:2rem">Model Info</p><p class="hero-subtitle">Technical details about our NLP pipeline and models</p></div>', unsafe_allow_html=True)

    st.markdown("### Best Model: SVM with TF-IDF")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Accuracy", "98.3%")
    with col2: st.metric("Precision", "99.2%")
    with col3: st.metric("Recall", "87.9%")
    with col4: st.metric("F1-Score", "93.2%")

    st.markdown("---")
    st.markdown("### Model Comparison Table")

    # Build data: try CSV first, fall back to hardcoded values
    try:
        results_df = pd.read_csv('data/model_comparison.csv', index_col=0)
        models = results_df.index.tolist()
        accuracies  = results_df['Accuracy'].tolist()
        precisions  = results_df['Precision'].tolist()
        recalls     = results_df['Recall'].tolist()
        f1s         = results_df['F1-Score'].tolist()
    except:
        models      = ['Naive Bayes (TF-IDF)', 'SVM (TF-IDF)', 'SVM (Word2Vec)']
        accuracies  = [0.9668, 0.9830, 0.9453]
        precisions  = [0.9912, 0.9924, 0.8729]
        recalls     = [0.7584, 0.8792, 0.6913]
        f1s         = [0.8593, 0.9324, 0.7715]

    # Identify best model by F1-Score
    best_idx = f1s.index(max(f1s))

    # Add BEST label to model name
    display_models = []
    for i, m in enumerate(models):
        display_models.append(m + "  ★ BEST" if i == best_idx else m)

    table_df = pd.DataFrame({
        'Model': display_models,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1s
    })

    def highlight_best(row):
        if '★ BEST' in str(row['Model']):
            return ['background-color: #1a5c2a; color: #ffffff; font-weight: bold'] * len(row)
        return ['background-color: #111111; color: #cccccc'] * len(row)

    styled = (
        table_df.style
        .apply(highlight_best, axis=1)
        .format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1-Score': '{:.4f}'})
        .set_properties(**{'text-align': 'right'})
        .set_properties(subset=['Model'], **{'text-align': 'left'})
        .set_table_styles([
            {'selector': 'thead th', 'props': [
                ('background-color', '#1a1a1a'),
                ('color', '#888888'),
                ('font-family', 'monospace'),
                ('font-size', '0.85rem'),
                ('text-transform', 'uppercase'),
                ('letter-spacing', '1px'),
                ('border-bottom', '2px solid #2a2a2a'),
                ('padding', '16px 20px'),
            ]},
            {'selector': 'td', 'props': [
                ('padding', '18px 20px'),
                ('border-bottom', '1px solid #1e1e1e'),
                ('font-family', 'monospace'),
                ('font-size', '1rem'),
            ]},
            {'selector': 'table', 'props': [
                ('border-collapse', 'collapse'),
                ('width', '100%'),
            ]},
        ])
        .hide(axis='index')
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=175)

    st.markdown("---")
    st.markdown("### NLP Pipeline")
    st.markdown("""
    | Step | Method | Description |
    |------|--------|-------------|
    | 1 | Lowercasing | Convert all text to lowercase |
    | 2 | URL Removal | Remove http/www links |
    | 3 | Special Char Removal | Remove numbers and punctuation |
    | 4 | Tokenization | Split text into individual words |
    | 5 | Stopword Removal | Remove common words (the, is, and) |
    | 6 | Lemmatization | Reduce words to their base form |
    """)

    st.markdown("---")
    st.markdown("### About the Models")
    with st.expander("Naive Bayes (TF-IDF)"):
        st.write("Naive Bayes is a probabilistic classifier based on Bayes' theorem. Works well with TF-IDF and is historically the most popular algorithm for spam filtering. Accuracy: 96.7% | F1: 85.9%")
    with st.expander("SVM with TF-IDF — Best Model"):
        st.write("SVM finds the optimal hyperplane separating spam from ham. Handles high-dimensional TF-IDF feature spaces very well and consistently outperforms Naive Bayes on text classification. Accuracy: 98.3% | F1: 93.2%")
    with st.expander("SVM with Word2Vec"):
        st.write("Uses averaged Word2Vec sentence embeddings as features. While Word2Vec captures semantic meaning, averaging vectors loses the specific keyword signals important for spam detection. Accuracy: 94.6% | F1: 77.4%")

    st.markdown("---")
    st.markdown("### Feature Extraction Methods")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("TF-IDF"):
            st.write("Converts text into numerical vectors by weighing word importance relative to the full dataset. Words like FREE and WIN score high in spam. Vocabulary: 5,000 features.")
    with col2:
        with st.expander("Word2Vec"):
            st.write("Neural word embeddings that capture semantic meaning. Each message is represented as the average of its 100-dimension word vectors. Vocabulary: 7,807 words.")

# ══════════════════════════════════════════════════════════
# PAGE 9 — HISTORY
# ══════════════════════════════════════════════════════════
elif page == "History":
    st.markdown('<div class="hero-banner"><p class="hero-title" style="font-size:2rem">Session History</p><p class="hero-subtitle">All messages analyzed in this session</p></div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div style="background:#0d0d0d; border:1px dashed #2a2a2a; border-radius:16px; padding:60px; text-align:center;">
            <div style="font-size:0.65rem; font-family:'IBM Plex Mono',monospace; text-transform:uppercase; letter-spacing:3px; color:#444444; margin-bottom:12px">[ NO DATA ]</div>
            <div style="font-size:0.9rem; color:#555555">No analysis history yet.<br>Start by analyzing a message in <strong style="color:#aaaaaa">Text Analyzer</strong>.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        total = len(st.session_state.history)
        spam_count = sum(1 for h in st.session_state.history if h['result'] == 'SPAM')
        ham_count = total - spam_count

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Analyzed", total)
        with col2: st.metric("Spam Found", spam_count)
        with col3: st.metric("Safe Found", ham_count)
        with col4:
            rate = f"{(spam_count/total*100):.1f}%" if total > 0 else "—"
            st.metric("Spam Rate", rate)

        st.markdown("---")

        col1, col2 = st.columns([2, 1])
        with col1:
            filter_result = st.selectbox("Filter by result:", ["All", "Spam only", "Ham only"])
        with col2:
            filter_source = st.selectbox("Filter by source:", ["All", "Text", "File", "Image", "Batch"])

        filtered = st.session_state.history.copy()
        if filter_result == "Spam only":
            filtered = [h for h in filtered if h['result'] == 'SPAM']
        elif filter_result == "Ham only":
            filtered = [h for h in filtered if h['result'] == 'HAM']
        if filter_source != "All":
            filtered = [h for h in filtered if h['source'] == filter_source]

        st.caption(f"Showing {len(filtered)} of {total} records")
        st.markdown("---")

        for i, item in enumerate(reversed(filtered)):
            css = "history-item" if item['result'] == 'SPAM' else "history-item ham"
            st.markdown(f"""
            <div class="{css}">
                <div class="history-msg"><strong>{item['result']}</strong> — {item['message']}</div>
                <div class="history-meta">
                    Confidence: {item['confidence']} &nbsp;&middot;&nbsp;
                    Trigger words: {item['keywords']} &nbsp;&middot;&nbsp;
                    Source: {item['source']} &nbsp;&middot;&nbsp;
                    #{total - i}
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        export_df = pd.DataFrame(st.session_state.history)
        st.download_button(
            "Download History as CSV",
            export_df.to_csv(index=False),
            "analysis_history.csv",
            "text/csv",
            use_container_width=True
        )

        if st.button("Clear All History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
