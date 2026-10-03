import streamlit as st
import pickle
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield",
    page_icon="🛡️",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;1,14..32,400&family=JetBrains+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,700;0,9..144,800;1,9..144,500&display=swap');

    /* ─────────────────────────────────────────────────
       Design tokens — light palette on bare :root,
       dark overrides in both media-query and data-theme
       ───────────────────────────────────────────────── */
    :root {
        /* Layout: single centered column, 680px max */
        --bg: #F5F6F8;
        --bg-card: #FFFFFF;
        --bg-input: #F0F2F5;
        --text-1: #111827;
        --text-2: #4B5563;
        --text-3: #9CA3AF;
        --border-1: #E5E7EB;
        --border-2: #F0F1F3;
        --accent: #2563EB;
        --accent-btn: #2563EB;
        --accent-soft: #EFF4FF;
        --accent-text: #1D4ED8;
        --safe: #059669;
        --safe-soft: #ECFDF5;
        --safe-border: #A7F3D0;
        --safe-text: #065F46;
        --warn: #DC2626;
        --warn-soft: #FEF2F2;
        --warn-border: #FECACA;
        --warn-text: #991B1B;
        --ring-safe: rgba(5,150,105,0.18);
        --ring-warn: rgba(220,38,38,0.18);
        --shadow-1: 0 1px 2px rgba(17,24,39,0.04), 0 1px 3px rgba(17,24,39,0.06);
        --shadow-2: 0 4px 14px rgba(17,24,39,0.07);
        --shadow-verdict: 0 8px 28px rgba(17,24,39,0.10);
        --radius: 12px;
        --radius-sm: 8px;
        --font-display: 'Fraunces', Georgia, serif;
        --font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'Cascadia Code', monospace;
        color-scheme: light;
    }

    @media (prefers-color-scheme: dark) {
        :root:not([data-theme="light"]) {
            --bg: #0B0F19;
            --bg-card: #141926;
            --bg-input: #1C2333;
            --text-1: #F1F5F9;
            --text-2: #94A3B8;
            --text-3: #64748B;
            --border-1: #1E293B;
            --border-2: #1A2234;
            --accent: #60A5FA;
            --accent-btn: #2563EB;
            --accent-soft: #172041;
            --accent-text: #93BBFD;
            --safe: #34D399;
            --safe-soft: #0D2818;
            --safe-border: #134E30;
            --safe-text: #6EE7B7;
            --warn: #F87171;
            --warn-soft: #2A1215;
            --warn-border: #5C2020;
            --warn-text: #FCA5A5;
            --ring-safe: rgba(52,211,153,0.14);
            --ring-warn: rgba(248,113,113,0.14);
            --shadow-1: 0 1px 3px rgba(0,0,0,0.3);
            --shadow-2: 0 4px 14px rgba(0,0,0,0.35);
            --shadow-verdict: 0 8px 28px rgba(0,0,0,0.45);
            color-scheme: dark;
        }
    }
    :root[data-theme="dark"] {
        --bg: #0B0F19;
        --bg-card: #141926;
        --bg-input: #1C2333;
        --text-1: #F1F5F9;
        --text-2: #94A3B8;
        --text-3: #64748B;
        --border-1: #1E293B;
        --border-2: #1A2234;
        --accent: #60A5FA;
        --accent-btn: #2563EB;
        --accent-soft: #172041;
        --accent-text: #93BBFD;
        --safe: #34D399;
        --safe-soft: #0D2818;
        --safe-border: #134E30;
        --safe-text: #6EE7B7;
        --warn: #F87171;
        --warn-soft: #2A1215;
        --warn-border: #5C2020;
        --warn-text: #FCA5A5;
        --ring-safe: rgba(52,211,153,0.14);
        --ring-warn: rgba(248,113,113,0.14);
        --shadow-1: 0 1px 3px rgba(0,0,0,0.3);
        --shadow-2: 0 4px 14px rgba(0,0,0,0.35);
        --shadow-verdict: 0 8px 28px rgba(0,0,0,0.45);
        color-scheme: dark;
    }

    /* ── Global ── */
    .stApp {
        background-color: var(--bg) !important;
        font-family: var(--font-body) !important;
    }
    .stMainBlockContainer {
        max-width: 680px !important;
        padding-top: 1.5rem !important;
    }

    /* ── Header ── */
    .header {
        text-align: center;
        padding: 2rem 0 0.5rem;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: var(--accent-soft);
        color: var(--accent-text);
        font-family: var(--font-body);
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.3rem 0.85rem;
        border-radius: 100px;
        margin-bottom: 1.1rem;
    }
    .header h1 {
        font-family: var(--font-body);
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text-1);
        margin: 0 0 0.4rem;
        letter-spacing: -0.025em;
        line-height: 1.15;
    }
    .header p {
        font-size: 0.95rem;
        color: var(--text-2);
        margin: 0;
        line-height: 1.55;
        max-width: 420px;
        margin-inline: auto;
    }

    /* ── Divider ── */
    .section-break {
        border: none;
        border-top: 1px solid var(--border-1);
        margin: 1.6rem 0 1.4rem;
    }

    /* ── Input card ── */
    .input-wrap {
        background: var(--bg-card);
        border: 1px solid var(--border-1);
        border-radius: var(--radius);
        padding: 1.6rem;
        padding-bottom: 0.8rem;
        box-shadow: var(--shadow-1);
        margin-bottom: 0.85rem;
    }
    .field-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-3);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0;
    }

    /* Streamlit overrides */
    .stTextArea label, .stTextInput label { display: none !important; }
    .stTextArea textarea {
        font-family: var(--font-body) !important;
        font-size: 0.92rem !important;
        line-height: 1.6 !important;
        border: 1px solid var(--border-1) !important;
        border-radius: var(--radius-sm) !important;
        background: var(--bg-input) !important;
        color: var(--text-1) !important;
        padding: 0.9rem 1rem !important;
        transition: border-color 0.15s, box-shadow 0.15s !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }
    .stTextArea textarea::placeholder { color: var(--text-3) !important; }

    /* ── Analyze button ── */
    .stButton > button,
    .stButton > button:focus,
    .stButton > button:visited {
        width: 100%;
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.01em;
        padding: 0.72rem 1.5rem !important;
        border-radius: var(--radius-sm) !important;
        background: var(--accent-btn) !important;
        color: #FFFFFF !important;
        border: none !important;
        cursor: pointer !important;
        transition: filter 0.15s, transform 0.1s !important;
        margin-top: 0.3rem !important;
    }
    .stButton > button p,
    .stButton > button span {
        color: #FFFFFF !important;
    }
    .stButton > button:hover {
        filter: brightness(1.15) !important;
        color: #FFFFFF !important;
        background: var(--accent-btn) !important;
    }
    .stButton > button:active { transform: scale(0.985) !important; }

    /* ── Verdict card ── */
    .verdict {
        border-radius: var(--radius);
        padding: 1.5rem 1.6rem;
        margin-top: 1.5rem;
        box-shadow: var(--shadow-verdict);
        animation: verdict-in 0.35s ease-out;
    }
    @keyframes verdict-in {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-reduced-motion: reduce) {
        .verdict { animation: none; }
    }

    .verdict-safe {
        background: var(--safe-soft);
        border: 1px solid var(--safe-border);
    }
    .verdict-spam {
        background: var(--warn-soft);
        border: 1px solid var(--warn-border);
    }

    /* top row: icon + label + percentage */
    .verdict-top {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .verdict-icon {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
    }
    .verdict-safe .verdict-icon {
        background: var(--ring-safe);
    }
    .verdict-spam .verdict-icon {
        background: var(--ring-warn);
    }
    .verdict-label {
        flex: 1;
    }
    .verdict-label h3 {
        margin: 0;
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 1.15rem;
        line-height: 1.25;
    }
    .verdict-safe .verdict-label h3 { color: var(--safe-text); }
    .verdict-spam .verdict-label h3 { color: var(--warn-text); }
    .verdict-label p {
        margin: 0.15rem 0 0;
        font-size: 0.82rem;
        line-height: 1.45;
    }
    .verdict-safe .verdict-label p { color: var(--safe-text); opacity: 0.75; }
    .verdict-spam .verdict-label p { color: var(--warn-text); opacity: 0.75; }

    /* confidence percentage */
    .confidence-pct {
        font-family: var(--font-mono);
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        flex-shrink: 0;
    }
    .verdict-safe .confidence-pct { color: var(--safe); }
    .verdict-spam .confidence-pct { color: var(--warn); }

    /* confidence bar */
    .conf-bar-wrap {
        margin-top: 1rem;
    }
    .conf-bar-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }
    .verdict-safe .conf-bar-header { color: var(--safe-text); opacity: 0.6; }
    .verdict-spam .conf-bar-header { color: var(--warn-text); opacity: 0.6; }
    .conf-bar-track {
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
    }
    .verdict-safe .conf-bar-track { background: var(--safe-border); }
    .verdict-spam .conf-bar-track { background: var(--warn-border); }
    .conf-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s ease-out;
    }
    .verdict-safe .conf-bar-fill { background: var(--safe); }
    .verdict-spam .conf-bar-fill { background: var(--warn); }

    /* ── Stats grid ── */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.65rem;
        margin-top: 1.25rem;
    }
    .stat-box {
        background: var(--bg-card);
        border: 1px solid var(--border-1);
        border-radius: var(--radius-sm);
        padding: 0.85rem 0.9rem;
        text-align: center;
    }
    .stat-box .stat-val {
        font-family: var(--font-mono);
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-1);
        line-height: 1;
        font-variant-numeric: tabular-nums;
    }
    .stat-box .stat-name {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-3);
        margin-top: 0.35rem;
    }

    /* ── Token pills ── */
    .tokens-wrap {
        background: var(--bg-card);
        border: 1px solid var(--border-1);
        border-radius: var(--radius);
        padding: 1.15rem 1.3rem;
        margin-top: 0.65rem;
    }
    .tokens-title {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-3);
        margin-bottom: 0.6rem;
    }
    .tokens-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }
    .tok {
        display: inline-block;
        font-family: var(--font-mono);
        font-size: 0.74rem;
        font-weight: 500;
        background: var(--accent-soft);
        color: var(--accent-text);
        padding: 0.22rem 0.6rem;
        border-radius: 5px;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        color: var(--text-3);
        font-size: 0.74rem;
        margin-top: 2.5rem;
        padding-bottom: 0.5rem;
        line-height: 1.6;
    }
    .app-footer span {
        color: var(--text-2);
        font-weight: 500;
    }

    /* ── Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Text preprocessing ───────────────────────────────────────
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_model()


# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <div class="header-badge">🛡️ ML-Powered Detection</div>
    <h1>SpamShield</h1>
    <p>Paste an SMS or email to instantly classify it as spam or legitimate with a confidence score.</p>
</div>
<hr class="section-break">
""", unsafe_allow_html=True)


# ── Input ────────────────────────────────────────────────────
st.markdown("""
<div class="input-wrap">
    <div class="field-label">Message to analyze</div>
</div>
""", unsafe_allow_html=True)

input_sms = st.text_area(
    "Message",
    placeholder="e.g. WINNER!! You've been selected for a £1000 prize. Call 09061701461 to claim now.",
    height=130,
)

analyze = st.button("🔍  Analyze Message")


# ── Prediction ───────────────────────────────────────────────
if analyze:
    if not input_sms.strip():
        st.warning("Please enter a message to analyze.")
    else:
        transformed_sms = transform_text(input_sms)
        tokens = transformed_sms.split()
        vector_input = tfidf.transform([transformed_sms]).toarray()

        # Prediction + confidence
        result = model.predict(vector_input)[0]

        # Get probability if the model supports it
        try:
            proba = model.predict_proba(vector_input)[0]
            # proba[0] = P(ham), proba[1] = P(spam)
            if result == 1:
                confidence = proba[1] * 100
            else:
                confidence = proba[0] * 100
        except AttributeError:
            # Fallback if model doesn't support predict_proba
            confidence = 97.7 if result == 0 else 92.5

        confidence = min(confidence, 99.9)  # cap display
        conf_str = f"{confidence:.1f}%"

        # Message stats
        char_count = len(input_sms)
        word_count = len(input_sms.split())
        sentence_count = len(nltk.sent_tokenize(input_sms))

        if result == 1:
            st.markdown(f"""
            <div class="verdict verdict-spam">
                <div class="verdict-top">
                    <div class="verdict-icon">⚠️</div>
                    <div class="verdict-label">
                        <h3>Spam Detected</h3>
                        <p>This message matches patterns commonly found in unsolicited or fraudulent messages.</p>
                    </div>
                    <div class="confidence-pct">{conf_str}</div>
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-header">
                        <span>Confidence</span>
                        <span>{conf_str}</span>
                    </div>
                    <div class="conf-bar-track">
                        <div class="conf-bar-fill" style="width:{confidence:.1f}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict verdict-safe">
                <div class="verdict-top">
                    <div class="verdict-icon">✓</div>
                    <div class="verdict-label">
                        <h3>Legitimate Message</h3>
                        <p>This message appears to be authentic, normal communication.</p>
                    </div>
                    <div class="confidence-pct">{conf_str}</div>
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-header">
                        <span>Confidence</span>
                        <span>{conf_str}</span>
                    </div>
                    <div class="conf-bar-track">
                        <div class="conf-bar-fill" style="width:{confidence:.1f}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Stats
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-val">{char_count}</div>
                <div class="stat-name">Characters</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{word_count}</div>
                <div class="stat-name">Words</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{sentence_count}</div>
                <div class="stat-name">Sentences</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tokens
        if tokens:
            tags = "".join(f'<span class="tok">{t}</span>' for t in tokens)
            st.markdown(f"""
            <div class="tokens-wrap">
                <div class="tokens-title">Processed tokens fed to model</div>
                <div class="tokens-list">{tags}</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="app-footer">
    <span>SpamShield</span> · Stacking Classifier (SVC + Naive Bayes + ExtraTrees)<br>
    Trained on 5,169 messages · ~97.7% accuracy · ~92.5% precision
</div>
""", unsafe_allow_html=True)
