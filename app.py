"""
app.py — News Checker  (Premium Edition)
Electric Blue & Deep Indigo theme with:
  • Animated hero + particle background
  • 2 input tabs: Paste / URL
  • Credibility signal analysis panel
  • Top suspicious / credible keyword highlights
  • Animated verdict card with pulse glow
  • Session timeline history with timestamps
  • Live character counter
Run with:  python -m streamlit run app.py
"""

import os, re, math, string, time, joblib, nltk, requests, collections
from datetime import datetime
from bs4 import BeautifulSoup
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

# ── NLTK ─────────────────────────────────────────────────────────────────────
for pkg in ["stopwords", "punkt", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fake_news_model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

# ── Fake-news signal words (for keyword highlight) ────────────────────────────
FAKE_SIGNALS = {
    "BREAKING","SHOCKING","EXPOSED","EXCLUSIVE","BOMBSHELL","SCANDAL",
    "CONSPIRACY","HOAX","FAKE","LIE","LIES","COVER","SUPPRESSED",
    "BANNED","LEAKED","SECRET","HIDDEN","TRUTH","WAKE","SHEEP",
    "MIRACLE","CURE","CANCER","CHIP","MICROCHIP","5G","ILLUMINATI",
    "DEEP","STATE","GLOBALIST","SATANIC","WHISTLEBLOWER","ELITE",
}
REAL_SIGNALS = {
    "according","reported","confirmed","announced","study","researchers",
    "scientists","government","official","statement","evidence","data",
    "survey","analysis","percent","statistics","published","reviewed",
    "journal","university","institute","ministry","agency","authority",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthGuard AI — Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Background ── */
.stApp {
    background:
        radial-gradient(ellipse 70% 50% at 20% 0%, rgba(59,130,246,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(99,102,241,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(15,23,62,0.8) 0%, transparent 100%),
        #070b1a;
    min-height: 100vh;
}

/* ── Animated grid overlay ── */
.stApp::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(59,130,246,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59,130,246,0.04) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: grid-drift 30s linear infinite;
}
@keyframes grid-drift {
    0%   { transform: translateY(0); }
    100% { transform: translateY(50px); }
}

/* ── Hero ── */
.hero-wrap { text-align:center; padding: 2.5rem 0 1rem; position:relative; z-index:1; }
.hero-shield {
    font-size: 3.5rem; margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 20px rgba(59,130,246,0.6));
    animation: shield-pulse 3s ease-in-out infinite alternate;
}
@keyframes shield-pulse {
    from { filter: drop-shadow(0 0 12px rgba(59,130,246,0.4)); transform: scale(1); }
    to   { filter: drop-shadow(0 0 30px rgba(99,102,241,0.8)); transform: scale(1.05); }
}
.hero-title {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 3.4rem; font-weight: 800;
    background: linear-gradient(135deg, #93c5fd 0%, #60a5fa 30%, #a855f7 60%, #93c5fd 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
    letter-spacing: -.03em; line-height: 1.05;
}
@keyframes shimmer { to { background-position: 200% center; } }
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem; margin-top: .6rem;
    background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.3);
    color: #93c5fd; font-size:.78rem; font-weight:600; letter-spacing:.08em;
    padding: .35rem 1rem; border-radius:30px;
    backdrop-filter: blur(10px);
}
.hero-sub {
    color: #475569; font-size: 1rem; margin-top:.6rem; letter-spacing: 0.01em;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(15,23,62,0.5);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 20px; padding: 1.6rem 1.8rem;
    margin-bottom: 1.1rem;
    backdrop-filter: blur(12px);
    transition: border-color .3s, box-shadow .3s, transform .2s;
}
.glass-card:hover {
    border-color: rgba(59,130,246,0.4);
    box-shadow: 0 8px 32px rgba(59,130,246,0.12);
    transform: translateY(-1px);
}

/* ── Verdict cards ── */
.verdict-real {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.06));
    border: 2px solid rgba(52,211,153,0.7); border-radius: 20px; padding: 2rem;
    text-align:center; animation: glowGreen 2.5s ease-in-out infinite alternate;
    backdrop-filter: blur(10px);
}
@keyframes glowGreen {
    from { box-shadow: 0 0 20px rgba(52,211,153,0.2); }
    to   { box-shadow: 0 0 45px rgba(52,211,153,0.5), 0 0 80px rgba(52,211,153,0.1); }
}
.verdict-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(185,28,28,0.06));
    border: 2px solid rgba(239,68,68,0.7); border-radius: 20px; padding: 2rem;
    text-align:center; animation: glowRed 2s ease-in-out infinite alternate;
    backdrop-filter: blur(10px);
}
@keyframes glowRed {
    from { box-shadow: 0 0 20px rgba(239,68,68,0.25); }
    to   { box-shadow: 0 0 45px rgba(239,68,68,0.6), 0 0 80px rgba(239,68,68,0.1); }
}
.verdict-label { font-size:2.5rem; font-weight:900; letter-spacing:.03em; color:#fff; font-family:'Space Grotesk',sans-serif; }
.verdict-conf  { font-size:1rem; color:#94a3b8; margin-top:.4rem; }

/* ── Stat pill ── */
.stat-pill {
    display:inline-flex; flex-direction:column; align-items:center;
    background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.18);
    border-radius:16px; padding:.9rem 1.2rem; min-width:90px;
    backdrop-filter: blur(8px);
    transition: border-color .2s, background .2s;
}
.stat-pill:hover { background: rgba(59,130,246,0.14); border-color: rgba(59,130,246,0.35); }
.stat-pill-num   { font-size:1.7rem; font-weight:800; color:#93c5fd; font-family:'Space Grotesk',sans-serif; }
.stat-pill-label { font-size:.7rem; color:#4b5563; margin-top:.2rem; letter-spacing:.08em; text-transform:uppercase; }

/* ── Signal badge ── */
.sig-fake { display:inline-block; background:rgba(239,68,68,0.15); color:#fca5a5;
    border:1px solid rgba(239,68,68,0.3); border-radius:8px;
    padding:.2rem .6rem; font-size:.78rem; font-weight:600; margin:3px; }
.sig-real { display:inline-block; background:rgba(52,211,153,0.12); color:#6ee7b7;
    border:1px solid rgba(52,211,153,0.25); border-radius:8px;
    padding:.2rem .6rem; font-size:.78rem; font-weight:600; margin:3px; }

/* ── Signal bar ── */
.signal-row { display:flex; align-items:center; gap:.7rem; margin:.5rem 0; }
.signal-label { color:#64748b; font-size:.82rem; width:170px; flex-shrink:0; }
.signal-bar-bg { flex:1; background:rgba(255,255,255,0.05); border-radius:20px; height:7px; overflow:hidden; }
.signal-bar-fill-danger { height:7px; border-radius:20px;
    background:linear-gradient(90deg,#be123c,#ef4444); transition:width .6s ease; }
.signal-bar-fill-ok     { height:7px; border-radius:20px;
    background:linear-gradient(90deg,#065f46,#34d399); transition:width .6s ease; }
.signal-val { font-size:.8rem; font-weight:700; width:36px; text-align:right; color:#94a3b8; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080e24 0%, #0d1435 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.15);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li { color: #64748b; font-size:.88rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(59,130,246,0.07); border-radius:14px; padding:5px; gap:4px;
    border: 1px solid rgba(59,130,246,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius:10px; color:#475569 !important; font-weight:600; padding:.5rem 1.4rem;
    background:transparent; transition: all .2s;
    font-family:'Inter',sans-serif;
}
.stTabs [data-baseweb="tab"]:hover { color:#94a3b8 !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#3b82f6,#4f46e5) !important;
    color:#fff !important; box-shadow:0 4px 20px rgba(59,130,246,0.4);
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display:none; }

/* ── Buttons ── */
.stButton > button {
    width:100%;
    background: linear-gradient(135deg,#1e40af,#3b82f6) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    padding:.75rem 1.6rem !important; font-size:.95rem !important; font-weight:700 !important;
    letter-spacing:.02em;
    transition: transform .15s, box-shadow .2s !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.3) !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(59,130,246,0.55) !important;
}
.stButton > button:active { transform:translateY(0) !important; }

/* ── Text area ── */
.stTextArea textarea {
    background: rgba(15,23,62,0.6) !important;
    border: 1.5px solid rgba(59,130,246,0.2) !important;
    border-radius:12px !important; color:#e2e8f0 !important; font-size:.94rem !important;
    caret-color:#60a5fa !important;
    backdrop-filter: blur(8px) !important;
}
.stTextArea textarea::placeholder { color:#374151 !important; }
.stTextArea textarea:focus {
    border-color: rgba(59,130,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}
.stTextArea label { color:#93c5fd !important; font-weight:600 !important; }

/* ── Text input ── */
.stTextInput input {
    background: rgba(15,23,62,0.6) !important;
    border:1.5px solid rgba(59,130,246,0.2) !important;
    border-radius:10px !important; color:#e2e8f0 !important;
    caret-color:#60a5fa !important;
}
.stTextInput input:focus {
    border-color: rgba(59,130,246,0.5) !important;
    box-shadow:0 0 0 3px rgba(59,130,246,0.12) !important;
}
.stTextInput label { color:#93c5fd !important; font-weight:600 !important; }

/* ── History item ── */
.hist-item {
    display:flex; align-items:center; gap:.8rem;
    background: rgba(15,23,62,0.4); border:1px solid rgba(99,102,241,0.1);
    border-radius:12px; padding:.7rem 1.1rem; margin-bottom:.45rem;
    transition: background .2s, border-color .2s;
    backdrop-filter: blur(6px);
}
.hist-item:hover { background:rgba(99,102,241,0.1); border-color:rgba(99,102,241,0.25); }
.hist-dot-fake { width:10px; height:10px; border-radius:50%; background:#ef4444; flex-shrink:0;
    box-shadow: 0 0 6px rgba(239,68,68,0.6); }
.hist-dot-real { width:10px; height:10px; border-radius:50%; background:#34d399; flex-shrink:0;
    box-shadow: 0 0 6px rgba(52,211,153,0.6); }
.hist-text { flex:1; color:#cbd5e1; font-size:.84rem; }
.hist-tag-fake { color:#fca5a5; font-weight:700; font-size:.82rem; }
.hist-tag-real { color:#6ee7b7; font-weight:700; font-size:.82rem; }
.hist-time { color:#374151; font-size:.75rem; }
.hist-conf { color:#4b5563; font-size:.8rem; }

/* ── Counter badge ── */
.char-counter { text-align:right; font-size:.78rem; color:#374151; margin-top:.25rem; }
.char-counter.warn { color:#f87171; }

/* ── Section heading ── */
.section-head {
    font-size:.72rem; font-weight:700; letter-spacing:.14em; color:#60a5fa;
    text-transform:uppercase; margin:.9rem 0 .6rem; display:flex; align-items:center; gap:.5rem;
}
.section-head::after {
    content:''; flex:1; height:1px; background:rgba(59,130,246,0.15);
}

/* ── Score ring (CSS only) ── */
.score-ring-wrap { display:flex; flex-direction:column; align-items:center; }
.score-ring {
    width:110px; height:110px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1.6rem; font-weight:900;
    border: 4px solid;
    position:relative;
    font-family: 'Space Grotesk', sans-serif;
}
.score-ring.fake { border-color:#ef4444; color:#f87171;
    box-shadow: 0 0 28px rgba(239,68,68,0.45), inset 0 0 14px rgba(239,68,68,0.1); }
.score-ring.real { border-color:#34d399; color:#34d399;
    box-shadow: 0 0 28px rgba(52,211,153,0.45), inset 0 0 14px rgba(52,211,153,0.08); }
.score-ring-label { font-size:.68rem; color:#374151; margin-top:.5rem; letter-spacing:.1em; text-transform:uppercase; }

/* ── Divider ── */
hr { border-color: rgba(192,21,42,0.1) !important; }

/* Scrollbar */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background: linear-gradient(#1e40af,#3b82f6); border-radius:3px; }
::-webkit-scrollbar-track { background:#070b1a; }

/* Expander */
[data-testid="stExpander"] { border:1px solid rgba(59,130,246,0.15) !important; border-radius:14px !important;
    background: rgba(15,23,62,0.3) !important; }
[data-testid="stExpander"] summary { color:#64748b !important; }
[data-testid="stExpander"] summary:hover { color:#93c5fd !important; }

/* ── Metrics / selectbox ── */
[data-testid="stMetric"] { background: rgba(15,23,62,0.4); border-radius:12px; padding:0.8rem; border:1px solid rgba(99,102,241,0.1); }

/* ── General ── */
h1,h2,h3,h4 { color:#e2e8f0 !important; font-family:'Space Grotesk','Inter',sans-serif !important; }
h1,h2,h3,h4 { color:#fff !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split()
              if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        return None, None
    try:
        return joblib.load(MODEL_PATH), joblib.load(VEC_PATH)
    except Exception:
        return None, None


def sigmoid(x):
    return 1 / (1 + math.exp(-abs(x)))


def predict(text: str):
    model, vectorizer = load_model()
    if model is None:
        return None
    cleaned   = clean_text(text)
    tfidf_vec = vectorizer.transform([cleaned])
    pred      = model.predict(tfidf_vec)[0]
    score     = model.decision_function(tfidf_vec)[0]
    conf      = round(sigmoid(score) * 100, 2)
    label     = "REAL" if pred == 1 else "FAKE"
    return {"label": label, "confidence": conf, "score": float(score)}

def predict_groq(text: str, groq_token: str):
    """Fallback / Secondary check using Groq API (LLaMA 3.3)."""
    try:
        from groq import Groq
        client = Groq(api_key=groq_token)
        # Limit text to 3000 chars to avoid overwhelming
        truncated_text = text[:3000]
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional fact-checker. Determine if the following news article is REAL or FAKE. Respond strictly in this format: [VERDICT] - [1 SHORT SENTENCE EXPLANATION]. The verdict MUST be either REAL or FAKE."},
                {"role": "user", "content": f"Is this news real or fake? \n\n{truncated_text}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0
        )
        response_text = chat_completion.choices[0].message.content.strip()
        
        # Parse [VERDICT] - [EXPLANATION]
        if "REAL" in response_text.upper()[:15]:
            lbl = "REAL"
        elif "FAKE" in response_text.upper()[:15]:
            lbl = "FAKE"
        else:
            lbl = "UNKNOWN"
            
        explanation = response_text
        if "-" in response_text:
            explanation = response_text.split("-", 1)[1].strip()
            
        return {"label": lbl, "explanation": explanation}
    except Exception as e:
        return {"error": str(e)}


# ── Credibility signals ───────────────────────────────────────────────────────
def compute_signals(text: str) -> dict:
    words   = text.split()
    n       = max(len(words), 1)
    upper   = sum(1 for w in words if w.isupper() and len(w) > 2)
    excl    = text.count("!")
    quest   = text.count("?")
    caps_r  = round(upper / n * 100, 1)

    # Fake / real keyword hits
    upper_words = {w.upper().strip(string.punctuation) for w in words}
    lower_words = {w.lower().strip(string.punctuation) for w in words}
    fake_hits = [w for w in upper_words if w in FAKE_SIGNALS]
    real_hits = [w for w in lower_words if w in REAL_SIGNALS]

    # Sentence avg length (long = formal = real, short breathless = fake)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sent_len = round(sum(len(s.split()) for s in sentences) / max(len(sentences), 1), 1)

    return {
        "word_count"   : len(words),
        "caps_ratio"   : caps_r,
        "exclamations" : excl,
        "questions"    : quest,
        "fake_keywords": fake_hits[:8],
        "real_keywords": real_hits[:8],
        "avg_sent_len" : avg_sent_len,
        "num_sentences": len(sentences),
    }


# ── AI-writing detector ─────────────────────────────────────────────────────
AI_PHRASES = [
    "it is worth noting", "it is important to note", "it should be noted",
    "furthermore", "moreover", "additionally", "in conclusion", "in summary",
    "to summarize", "in other words", "on the other hand", "it is clear that",
    "it can be seen that", "needless to say", "as mentioned earlier",
    "as previously stated", "in light of", "with regard to", "with respect to",
    "it goes without saying", "at the end of the day", "in today's world",
    "in the modern era", "due to the fact that", "it is essential to",
    "a wide range of", "a variety of", "in terms of", "plays a crucial role",
    "delve into", "dive into", "let's explore", "let us explore",
    "comprehensive overview", "overall", "it's important to remember",
]
PERSONAL_PRONOUNS = {"i","me","my","mine","myself","we","us","our","ours",
                     "ourselves","you","your","yours"}
PASSIVE_MARKERS  = [" is "," are "," was "," were "," be "," been "," being "]


def detect_ai_writing(text: str) -> dict:
    """
    Heuristic AI-writing score (0-100).
    Higher = more likely AI-generated.
    """
    tl  = text.lower()
    words    = text.split()
    n_words  = max(len(words), 1)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_sent    = max(len(sentences), 1)
    sent_lens = [len(s.split()) for s in sentences]

    # 1. Burstiness: low std-dev of sentence lengths = AI trait
    if n_sent > 2:
        mean_sl = sum(sent_lens) / n_sent
        variance = sum((l - mean_sl) ** 2 for l in sent_lens) / n_sent
        std_dev  = variance ** 0.5
        # normalise: std_dev < 4 = very uniform (AI); > 12 = very human
        burstiness_score = max(0, min(100, (12 - std_dev) / 12 * 100))
    else:
        burstiness_score = 50.0

    # 2. AI phrase density
    ai_hits = [p for p in AI_PHRASES if p in tl]
    ai_phrase_score = min(100, len(ai_hits) / max(n_words / 100, 1) * 180)

    # 3. Lack of personal pronouns
    lower_words  = [w.lower().strip(string.punctuation) for w in words]
    pron_count   = sum(1 for w in lower_words if w in PERSONAL_PRONOUNS)
    pron_density = pron_count / n_words * 100
    # < 0.5% = very AI-like (formal/impersonal); > 3% = very human
    pronoun_score = max(0, min(100, (1.5 - pron_density) / 1.5 * 100))

    # 4. Passive voice ratio
    passive_hits = sum(1 for m in PASSIVE_MARKERS if m in tl)
    passive_score = min(100, passive_hits / max(n_sent / 5, 1) * 40)

    # 5. Low informal punctuation (em-dashes, ellipses, parens = human)
    informal_chars = text.count('—') + text.count('–') + text.count('…') + \
                     text.count('(') + text.count(')') + text.count('...')
    informal_density = informal_chars / n_words * 100
    informal_score = max(0, min(100, (2 - informal_density) / 2 * 100))

    # 6. Transition word density
    transitions = ["however","therefore","thus","hence","consequently",
                   "subsequently","nevertheless","nonetheless","meanwhile"]
    trans_hits  = sum(1 for t in transitions if t in tl)
    trans_score = min(100, trans_hits / max(n_words / 80, 1) * 70)

    # Weighted average
    ai_score = round(
        burstiness_score  * 0.28 +
        ai_phrase_score   * 0.28 +
        pronoun_score     * 0.18 +
        passive_score     * 0.10 +
        informal_score    * 0.10 +
        trans_score       * 0.06,
    1)

    if ai_score >= 65:
        label, color = "Likely AI-Written",    "#a78bfa"
    elif ai_score >= 40:
        label, color = "Possibly AI-Written",  "#fbbf24"
    else:
        label, color = "Likely Human-Written", "#34d399"

    return {
        "score"          : ai_score,
        "label"          : label,
        "color"          : color,
        "burstiness"     : round(burstiness_score, 1),
        "ai_phrase_pct"  : round(ai_phrase_score, 1),
        "pronoun_score"  : round(pronoun_score, 1),
        "passive_score"  : round(passive_score, 1),
        "informal_score" : round(informal_score, 1),
        "ai_hits"        : ai_hits[:6],
        "pron_density"   : round(pron_density, 2),
    }


def render_signal_bar(label, value, max_val, danger=True):
    pct   = min(int(value / max(max_val, 1) * 100), 100)
    cls   = "signal-bar-fill-danger" if danger else "signal-bar-fill-ok"
    color = "#ff4d6d" if danger else "#34d399"
    st.markdown(
        f"<div class='signal-row'>"
        f"  <span class='signal-label'>{label}</span>"
        f"  <div class='signal-bar-bg'><div class='{cls}' style='width:{pct}%;'></div></div>"
        f"  <span class='signal-val' style='color:{color}'>{value}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def make_gauge(confidence, label):
    color = "#34d399" if label == "REAL" else "#c0152a"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"color": color, "size": 38, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#2a2a2a",
                     "tickwidth": 1, "tickfont": {"color": "#555"}},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,.05)",
            "steps": [
                {"range": [0,  40], "color": "rgba(192,21,42,.08)"},
                {"range": [40, 70], "color": "rgba(251,191,36,.06)"},
                {"range": [70,100], "color": "rgba(52,211,153,.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.78, "value": confidence},
        },
    ))
    fig.update_layout(
        height=220, margin=dict(t=14, b=8, l=16, r=16),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
    )
    return fig


def scrape_article(url: str) -> dict:
    """
    Robust article scraper with 4-layer fallback:
    1. BeautifulSoup + full browser headers + 20 article-body CSS selectors
    2. Open Graph / Twitter / description meta tags (partial but usable)
    3. newspaper3k (handles many anti-scraper sites)
    4. Google AMP variant of the URL
    """
    # Full browser-mimicking headers
    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language":           "en-US,en;q=0.9",
        "Accept-Encoding":           "gzip, deflate",
        "Connection":                "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control":             "max-age=0",
        "Sec-Fetch-Dest":            "document",
        "Sec-Fetch-Mode":            "navigate",
        "Sec-Fetch-Site":            "none",
        "Sec-Fetch-User":            "?1",
        "DNT":                       "1",
    }

    if not url.startswith(("http://", "https://")):
        return {"title": "", "text": "",
                "error": "Invalid URL — must start with http:// or https://"}

    def _get(u):
        try:
            r = requests.get(u, headers=hdrs, timeout=15, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception:
            return None

    def _extract(soup):
        """Extract title + body from a BeautifulSoup object."""
        # Title
        title = ""
        og_t  = soup.find("meta", property="og:title")
        if og_t: title = og_t.get("content", "").strip()
        if not title:
            for sel in ["h1","[class*='headline']","[class*='article-title']",
                        "[class*='story-title']","[class*='title']"]:
                tag = soup.select_one(sel)
                if tag: title = tag.get_text(strip=True); break
        if not title and soup.title:
            title = soup.title.get_text(strip=True)

        # Body — try 20 selectors before fallback
        body = ""
        for sel in [
            "article",
            "[class*='article-body']", "[class*='articleBody']",
            "[class*='story-body']",   "[class*='storyBody']",
            "[class*='post-content']", "[class*='postContent']",
            "[class*='entry-content']","[class*='entryContent']",
            "[class*='news-content']", "[class*='newsContent']",
            "[class*='article-text']", "[class*='full-article']",
            "[class*='story-content']","[class*='content-area']",
            "[id*='article-body']",    "[id*='storyContent']",
            "[itemprop='articleBody']","main",
        ]:
            c = soup.select_one(sel)
            if c:
                candidate = " ".join(
                    p.get_text(strip=True) for p in c.find_all("p")
                    if len(p.get_text(strip=True)) > 25
                )
                if len(candidate) > 200:
                    body = candidate
                    break

        # Fallback: all <p> tags on page
        if len(body) < 200:
            body = " ".join(
                p.get_text(strip=True) for p in soup.find_all("p")
                if len(p.get_text(strip=True)) > 25
            )
        body = re.sub(r"\s+", " ", body).strip()
        return title, body

    def _meta_text(soup):
        """Pull content from OG/Twitter/description meta tags."""
        parts = []
        for prop in ["og:description", "twitter:description"]:
            t = soup.find("meta", property=prop)
            if t: parts.append(t.get("content","").strip())
        for name in ["description", "twitter:description"]:
            t = soup.find("meta", attrs={"name": name})
            if t: parts.append(t.get("content","").strip())
        return " ".join(dict.fromkeys(p for p in parts if p))  # deduplicate

    # ── Layer 1: BeautifulSoup with full headers ──────────────────────────
    resp = _get(url)
    if resp is not None:
        soup = BeautifulSoup(resp.text, "lxml")
        title, body = _extract(soup)
        if len(body) >= 150:
            return {"title": title,
                    "text": (f"{title}. " if title else "") + body,
                    "error": ""}

        # ── Layer 1b: meta description fallback ──────────────────────────
        og_title = (soup.find("meta", property="og:title") or {}).get("content","").strip()
        if og_title and not title: title = og_title
        meta_body = _meta_text(soup)
        if len(meta_body) >= 60:
            return {
                "title"  : title,
                "text"   : (f"{title}. " if title else "") + meta_body,
                "error"  : "",
                "warning": (
                    "⚠️ Only the article summary was extracted (the site blocks full scraping). "
                    "For better analysis, paste the full article text in the **Paste Text** tab."
                ),
            }

    # ── Layer 2: newspaper3k ─────────────────────────────────────────────
    try:
        from newspaper import Article as NpArt
        art = NpArt(url, browser_user_agent=hdrs["User-Agent"], request_timeout=15)
        art.download()
        art.parse()
        if len(art.text or "") >= 150:
            t = art.title or ""
            return {"title": t,
                    "text": (f"{t}. " if t else "") + art.text,
                    "error": ""}
    except Exception:
        pass

    # ── Layer 3: Google AMP variant ──────────────────────────────────────
    try:
        from urllib.parse import urlparse, urlunparse
        p   = urlparse(url)
        amp = urlunparse(p._replace(path="/amp" + p.path))
        ar  = _get(amp)
        if ar:
            soup2 = BeautifulSoup(ar.text, "lxml")
            t2, b2 = _extract(soup2)
            if len(b2) >= 150:
                return {"title": t2,
                        "text": (f"{t2}. " if t2 else "") + b2,
                        "error": ""}
    except Exception:
        pass

    # ── All layers failed ────────────────────────────────────────────────
    return {
        "title": "", "text": "",
        "error": (
            "Could not extract article text from this URL. "
            "The site likely blocks automated access or requires JavaScript.\n\n"
            "**What you can do:**\n"
            "1. Open the article in your browser\n"
            "2. Select all text (Ctrl+A), copy (Ctrl+C)\n"
            "3. Switch to the **📝 Paste Text** tab and paste it there"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for key, val in [("history",[]),("total",0),("fake_n",0),("real_n",0)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 .5rem;'>
      <div style='font-size:2.8rem;filter:drop-shadow(0 0 14px rgba(99,102,241,0.7));'>🛡️</div>
      <div style='font-size:1.25rem;font-weight:800;background:linear-gradient(135deg,#a5b4fc,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-top:0.3rem;font-family:Space Grotesk,Inter,sans-serif;'>TruthGuard AI</div>
      <div style='font-size:.68rem;color:#374151;letter-spacing:.14em;margin-top:0.2rem;text-transform:uppercase;'>Intelligence Engine v2.5</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Model status
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        st.markdown("""<div style='background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.3);
        border-radius:10px;padding:.6rem 1rem;color:#34d399;font-size:.85rem;font-weight:600;text-align:center;'>
        ✅ AI Model Ready</div>""", unsafe_allow_html=True)
    else:
        st.error("❌ Model not trained")
        if st.button("🚀 Train Model Now", help="Click to train the AI model on sample data"):
            with st.spinner("Training model... please wait..."):
                try:
                    import subprocess
                    # Run the training script
                    result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ Model trained successfully!")
                        st.rerun()
                    else:
                        st.error(f"Training failed: {result.stderr}")
                except Exception as e:
                    st.error(f"Error starting trainer: {e}")

    st.markdown("---")

    # Groq Extension
    groq_token = os.getenv("GROQ_API_KEY", "")
    if not groq_token:
        st.markdown("<div class='section-head'>Deep Learning Backup</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:.78rem;color:#777;margin-bottom:.5rem;'>"
                    "Use a free Groq API token to cross-check results with LLaMA 3.3.</div>", 
                    unsafe_allow_html=True)
        groq_token = st.text_input("🔑 Groq Token (Optional)", type="password", 
                                 placeholder="gsk_xxxxxxxxxxxxx", help="Get one for free at console.groq.com")
    
    st.markdown("---")

    # Live session stats
    st.markdown("<div class='section-head'>Session Stats</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, num, lbl, clr in [
        (c1, st.session_state.total,  "Total",  "#ff4d6d"),
        (c2, st.session_state.real_n, "Real",   "#34d399"),
        (c3, st.session_state.fake_n, "Fake",   "#c0152a"),
    ]:
        col.markdown(
            f"<div class='stat-pill'>"
            f"<span class='stat-pill-num' style='color:{clr}'>{num}</span>"
            f"<span class='stat-pill-label'>{lbl}</span></div>",
            unsafe_allow_html=True,
        )

    if st.session_state.total > 0:
        fake_pct = round(st.session_state.fake_n / st.session_state.total * 100)
        st.markdown(
            f"<div style='margin:.9rem 0 .2rem;font-size:.75rem;color:#555;'>Fake rate this session</div>"
            f"<div style='background:rgba(255,255,255,.05);border-radius:20px;height:6px;'>"
            f"<div style='width:{fake_pct}%;height:6px;border-radius:20px;"
            f"background:linear-gradient(90deg,#1e40af,#3b82f6);'></div></div>"
            f"<div style='color:#3b82f6;font-size:.78rem;font-weight:700;text-align:right;margin-top:.2rem;'>{fake_pct}%</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("<div class='section-head'>How it works</div>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:.82rem;color:#555;line-height:1.7;'>
• <b style='color:#888'>TF-IDF</b> vectorises text into 50 000 features<br>
• <b style='color:#888'>PassiveAggressiveClassifier</b> classifies in real-time<br>
• Trained on <b style='color:#888'>ISOT Fake News Dataset</b><br>
• Accuracy <b style='color:#ff4d6d'>~93–96%</b> on full dataset
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-head'>Tips</div>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:.8rem;color:#555;line-height:1.8;'>
💡 Longer articles = better accuracy<br>
💡 Paste full article, not just headline<br>
💡 Works with any public news URL
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero-wrap'>
  <div class='hero-shield'>🛡️</div>
  <div class='hero-title'>TruthGuard AI</div>
  <div style='margin-top:.5rem;'>
    <span class='hero-badge'>🔬 AI-Powered &nbsp;·&nbsp; 🧠 LLaMA 3.3 Verified &nbsp;·&nbsp; ⚡ Real-time</span>
  </div>
  <div class='hero-sub' style='margin-top:.7rem;'>
    Advanced fake news detection powered by Machine Learning + Large Language Models
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INPUT TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_text, tab_url = st.tabs([
    "📝  Paste Text",
    "🔗  Article URL",
])

article_text = ""
analyze_btn  = False
source_label = ""  # for history preview

# ───────────────────────── TAB 1: Paste Text ─────────────────────────────────
with tab_text:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    raw_txt = st.text_area(
        "📰 Paste your news article or headline below",
        height=200,
        placeholder="Enter or paste the full news article text here…",
        key="txt_input",
    )
    # live char counter
    wc = len(raw_txt.split()) if raw_txt else 0
    cc = len(raw_txt)
    counter_cls = "char-counter warn" if wc < 20 and raw_txt else "char-counter"
    st.markdown(
        f"<div class='{counter_cls}'>{wc} words · {cc} chars"
        + (" — add more text for better accuracy" if wc < 20 and raw_txt else "") + "</div>",
        unsafe_allow_html=True,
    )
    bc1, bc2, bc3 = st.columns([3, 1, 1])
    with bc1: analyse_txt_btn = st.button("🔍 Analyse Article", use_container_width=True, key="btn_txt")
    with bc2: clear_btn       = st.button("🗑 Clear",           use_container_width=True, key="btn_clr")
    with bc3: hist_btn        = st.button("📋 History",         use_container_width=True, key="btn_hst")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("💡 Try example articles"):
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("<span style='color:#ff4d6d;font-weight:700;'>🚨 Likely FAKE</span>", unsafe_allow_html=True)
            st.code("SHOCKING: Scientists secretly CONFIRM lemon water cures ALL cancers overnight — Big Pharma doesn't want you to know!", language="text")
        with ec2:
            st.markdown("<span style='color:#34d399;font-weight:700;'>✅ Likely REAL</span>", unsafe_allow_html=True)
            st.code("The Federal Reserve raised interest rates by 25 basis points, citing continued progress on inflation while signalling caution over future hikes.", language="text")

    if analyse_txt_btn and raw_txt.strip():
        article_text = raw_txt.strip()
        source_label = raw_txt[:60] + "…"
        analyze_btn  = True
    elif analyse_txt_btn:
        st.warning("⚠️ Please paste some text first.")

# ───────────────────────── TAB 2: Article URL ────────────────────────────────
with tab_url:
    st.markdown(
        "<p style='color:#666;font-size:.9rem;margin-bottom:.8rem;'>"
        "Paste any news article link — we'll fetch and analyse it automatically.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    url_in = st.text_input("🔗 Article URL", placeholder="https://www.bbc.com/news/…", key="url_in")
    uc1, uc2 = st.columns([3, 1])
    with uc1: fetch_url_btn  = st.button("🔍 Fetch & Analyse", use_container_width=True, key="btn_url")
    with uc2: hist_btn_url   = st.button("📋 History",          use_container_width=True, key="btn_url_h")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🌐 Recommended sources"):
        st.markdown("""
| Source | URL |
|--------|-----|
| BBC News | `https://www.bbc.com/news` |
| Reuters | `https://www.reuters.com` |
| AP News | `https://apnews.com` |
| The Guardian | `https://www.theguardian.com` |
| Al Jazeera | `https://www.aljazeera.com` |
| CNN | `https://edition.cnn.com` |
| NDTV | `https://www.ndtv.com` |
| Times of India | `https://timesofindia.indiatimes.com` |
| The Hindu | `https://www.thehindu.com` |
        """)

    if fetch_url_btn:
        if not url_in.strip():
            st.warning("\u26a0\ufe0f Please enter a URL.")
        else:
            with st.spinner("\U0001f517 Fetching article\u2026 (trying multiple methods)"):
                scraped = scrape_article(url_in.strip())
            if scraped["error"]:
                st.error(scraped["error"])
            else:
                article_text = scraped["text"]
                source_label = scraped["title"][:60] or url_in[:60]
                analyze_btn  = True
                if scraped.get("warning"):
                    st.warning(scraped["warning"])
                st.markdown(
                    f"<div class='glass-card' style='max-height:140px;overflow-y:auto;font-size:.83rem;color:#bbb;'>"
                    f"<b style='color:#ff4d6d;'>{scraped['title']}</b><br><br>"
                    + scraped["text"][:600] + ("\u2026" if len(scraped["text"]) > 600 else "")
                    + "</div>", unsafe_allow_html=True,
                )



# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS & RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if analyze_btn:
    if not article_text.strip():
        st.warning("⚠️ No text to analyse.")
    elif not model:
        st.error("❌ Model not loaded. Run `python train_model.py` first.")
    else:
        with st.spinner("🧠 Analysing with AI…"):
            result   = predict(article_text)
            signals  = compute_signals(article_text)
            ai_det   = detect_ai_writing(article_text)
            groq_res   = None
            if groq_token:
                groq_res = predict_groq(article_text, groq_token)
            time.sleep(0.3)   # brief pause for UX drama

        if result:
            st.markdown("---")
            label = result["label"]
            conf  = result["confidence"]

            # ── Row 1: Verdict + Gauge + Score ring ──────────────────────────
            v_col, g_col, r_col = st.columns([1.8, 1.4, 1])

            with v_col:
                if label == "REAL":
                    st.markdown(f"""
                    <div class='verdict-real'>
                      <div class='verdict-label'>✅ REAL NEWS</div>
                      <div class='verdict-conf'>Confidence: {conf:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='verdict-fake'>
                      <div class='verdict-label'>🚨 FAKE NEWS</div>
                      <div class='verdict-conf'>Confidence: {conf:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                # Confidence interpretation
                st.markdown("<br>", unsafe_allow_html=True)
                if conf >= 92:
                    st.markdown("🔒 **Very high confidence** — The AI is extremely certain.", unsafe_allow_html=False)
                elif conf >= 78:
                    st.markdown("🔍 **High confidence** — Result is likely correct.", unsafe_allow_html=False)
                elif conf >= 63:
                    st.markdown("⚠️ **Moderate confidence** — Cross-check recommended.", unsafe_allow_html=False)
                else:
                    st.markdown("❓ **Low confidence** — Please verify with trusted sources.", unsafe_allow_html=False)

            with g_col:
                st.plotly_chart(make_gauge(conf, label), use_container_width=True)

            with r_col:
                ring_cls = "fake" if label == "FAKE" else "real"
                st.markdown(
                    f"<div class='score-ring-wrap' style='margin-top:1.2rem;'>"
                    f"<div class='score-ring {ring_cls}'>{conf:.0f}%</div>"
                    f"<div class='score-ring-label'>CONFIDENCE</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                # Word count pill
                st.markdown(
                    f"<div style='text-align:center;margin-top:.8rem;'>"
                    f"<div class='stat-pill' style='display:inline-flex;'>"
                    f"<span class='stat-pill-num' style='font-size:1.2rem;'>{signals['word_count']}</span>"
                    f"<span class='stat-pill-label'>WORDS</span></div></div>",
                    unsafe_allow_html=True,
                )

            # ── Row 2: Credibility Signals + Keyword Highlights ───────────────
            st.markdown("<br>", unsafe_allow_html=True)
            sig_col, kw_col = st.columns([1, 1])

            with sig_col:
                st.markdown("<div class='section-head'>Credibility Signals</div>", unsafe_allow_html=True)
                render_signal_bar("ALL-CAPS words",    signals["caps_ratio"],   60,  danger=True)
                render_signal_bar("Exclamation marks", signals["exclamations"], 10,  danger=True)
                render_signal_bar("Question marks",    signals["questions"],    8,   danger=True)
                render_signal_bar("Avg sentence length (words)", signals["avg_sent_len"], 40, danger=False)
                render_signal_bar("Number of sentences", signals["num_sentences"], 60, danger=False)

            with kw_col:
                st.markdown("<div class='section-head'>Keyword Analysis</div>", unsafe_allow_html=True)

                if signals["fake_keywords"]:
                    st.markdown("**🚨 Suspicious keywords detected:**", unsafe_allow_html=False)
                    badges = " ".join(f"<span class='sig-fake'>{w}</span>" for w in signals["fake_keywords"])
                    st.markdown(f"<div style='margin:.3rem 0 .8rem;'>{badges}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#555;font-size:.85rem;'>No suspicious keywords found</span>", unsafe_allow_html=True)

                if signals["real_keywords"]:
                    st.markdown("**✅ Credible language detected:**", unsafe_allow_html=False)
                    badges = " ".join(f"<span class='sig-real'>{w}</span>" for w in signals["real_keywords"])
                    st.markdown(f"<div style='margin:.3rem 0;'>{badges}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#555;font-size:.85rem;'>No credible anchor words found</span>", unsafe_allow_html=True)

                # Quick-tips
                st.markdown("<br>", unsafe_allow_html=True)
                tips = []
                if signals["caps_ratio"] > 15:
                    tips.append("High ALL-CAPS usage is a strong fake-news signal.")
                if signals["exclamations"] > 3:
                    tips.append("Excessive exclamation marks indicate sensationalism.")
                if signals["avg_sent_len"] < 10 and signals["num_sentences"] > 5:
                    tips.append("Short, choppy sentences are common in misinformation.")
                if signals["avg_sent_len"] > 25:
                    tips.append("Long, formal sentences suggest journalistic writing.")
                for tip in tips[:2]:
                    st.markdown(f"<div style='background:rgba(192,21,42,.07);border-left:3px solid #c0152a;"
                                f"padding:.5rem .8rem;border-radius:0 8px 8px 0;font-size:.82rem;color:#aaa;margin-bottom:.4rem;'>"
                                f"💡 {tip}</div>", unsafe_allow_html=True)

            # ── Row 3: Groq LLM Verification (if enabled) ────────────────
            if groq_token:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='section-head'>LLaMA 3.3 Verification (Groq)</div>", unsafe_allow_html=True)
                if groq_res and not groq_res.get("error") and groq_res.get("label") != "UNKNOWN":
                    g_lbl = groq_res['label']
                    g_exp = groq_res.get('explanation', '')
                    g_ring_cls = "fake" if g_lbl == "FAKE" else "real"
                    g_color = "#ff4d6d" if g_lbl == "FAKE" else "#34d399"
                    
                    st.markdown(
                        f"<div class='glass-card' style='display:flex;align-items:center;padding:1rem;gap:1.5rem;'>"
                        f"  <div class='score-ring {g_ring_cls}' style='width:60px;height:60px;font-size:.9rem;'>"
                        f"  LLaMA</div>"
                        f"  <div>"
                        f"    <div style='font-size:1.1rem;font-weight:700;color:{g_color};'>Verdict: {g_lbl}</div>"
                        f"    <div style='font-size:.85rem;color:#bbb;margin-top:0.3rem;'><b>Explanation:</b> {g_exp}</div>"
                        f"  </div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                elif groq_res and groq_res.get("error"):
                    st.warning(f"⚠️ Groq API Error: {groq_res['error']}")
                else:
                    st.warning("⚠️ Groq API returned an unknown response.")

            # ── Row 4: AI-Writing Detection ───────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-head'>AI-Writing Detection</div>", unsafe_allow_html=True)

            ai_col_ring, ai_col_bars, ai_col_phrases = st.columns([1, 1.4, 1.4])

            with ai_col_ring:
                ai_s   = ai_det["score"]
                ai_clr = ai_det["color"]
                ai_lbl = ai_det["label"]
                ring_border = ai_clr
                st.markdown(
                    f"<div style='display:flex;flex-direction:column;align-items:center;margin-top:.5rem;'>"
                    f"<div style='width:100px;height:100px;border-radius:50%;"
                    f"border:5px solid {ring_border};"
                    f"display:flex;align-items:center;justify-content:center;"
                    f"font-size:1.5rem;font-weight:900;color:{ai_clr};"
                    f"box-shadow:0 0 24px {ai_clr}55;'>{ai_s:.0f}%</div>"
                    f"<div style='margin-top:.5rem;font-size:.72rem;color:#555;letter-spacing:.06em;'>AI SCORE</div>"
                    f"<div style='margin-top:.6rem;background:{ai_clr}22;border:1px solid {ai_clr}55;"
                    f"border-radius:20px;padding:.25rem .8rem;font-size:.78rem;font-weight:700;color:{ai_clr};'>"
                    f"{ai_lbl}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with ai_col_bars:
                def ai_bar(lbl, val, tip=""):
                    pct = min(int(val), 100)
                    clr = ai_det["color"]
                    st.markdown(
                        f"<div class='signal-row'>"
                        f"  <span class='signal-label' title='{tip}'>{lbl}</span>"
                        f"  <div class='signal-bar-bg'>"
                        f"    <div style='width:{pct}%;height:8px;border-radius:20px;"
                        f"background:linear-gradient(90deg,{clr}88,{clr});'></div></div>"
                        f"  <span class='signal-val' style='color:{clr}'>{pct}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                ai_bar("Uniform sentence length", ai_det["burstiness"],    "Low variance = AI trait")
                ai_bar("AI phrase density",        ai_det["ai_phrase_pct"], "Common ChatGPT/Claude phrases")
                ai_bar("Impersonal language",      ai_det["pronoun_score"], "Low personal pronoun use")
                ai_bar("Passive voice usage",      ai_det["passive_score"], "AI prefers passive constructions")
                ai_bar("Lack of informal style",   ai_det["informal_score"],"Em-dashes, ellipses, parens")

            with ai_col_phrases:
                st.markdown(
                    f"<div style='font-size:.78rem;color:#666;margin-bottom:.4rem;'>"
                    f"Personal pronoun density: "
                    f"<b style='color:{ai_det['color']}'>{ai_det['pron_density']}%</b></div>",
                    unsafe_allow_html=True,
                )
                if ai_det["ai_hits"]:
                    st.markdown("<div style='font-size:.8rem;color:#888;margin-bottom:.3rem;'>AI phrases detected:</div>",
                                unsafe_allow_html=True)
                    for phrase in ai_det["ai_hits"]:
                        st.markdown(
                            f"<div style='background:rgba(167,139,250,.1);border:1px solid rgba(167,139,250,.25);"
                            f"border-radius:6px;padding:.2rem .6rem;font-size:.78rem;color:#a78bfa;"
                            f"margin-bottom:.3rem;display:inline-block;margin-right:.3rem;'>\"…{phrase}…\"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        "<div style='color:#555;font-size:.82rem;'>No common AI phrases detected.</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"<div style='margin-top:.8rem;background:rgba(167,139,250,.06);"
                    f"border-left:3px solid #a78bfa;padding:.5rem .8rem;"
                    f"border-radius:0 8px 8px 0;font-size:.78rem;color:#888;'>"
                    f"⚠️ Heuristic detection only — not 100% accurate. "
                    f"Always apply human judgment.</div>",
                    unsafe_allow_html=True,
                )

            # ── Update session state ──────────────────────────────────────────
            st.session_state.total += 1
            if label == "FAKE":
                st.session_state.fake_n += 1
            else:
                st.session_state.real_n += 1

            ts = datetime.now().strftime("%H:%M")
            st.session_state.history.append({
                "preview"   : source_label or article_text[:70] + "…",
                "label"     : label,
                "confidence": conf,
                "time"      : ts,
                "caps"      : signals["caps_ratio"],
                "fake_kw"   : len(signals["fake_keywords"]),
            })


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY PANEL
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.history:
    st.markdown("---")
    st.markdown("<div class='section-head'>Recent Checks</div>", unsafe_allow_html=True)

    for item in reversed(st.session_state.history[-12:]):
        dot_cls = "hist-dot-fake" if item["label"] == "FAKE" else "hist-dot-real"
        tag_cls = "hist-tag-fake" if item["label"] == "FAKE" else "hist-tag-real"
        st.markdown(
            f"<div class='hist-item'>"
            f"  <div class='{dot_cls}'></div>"
            f"  <div class='hist-text'>{item['preview']}</div>"
            f"  <div class='{tag_cls}'>{item['label']}</div>"
            f"  <div class='hist-conf'>{item['confidence']:.0f}%</div>"
            f"  <div class='hist-time'>{item['time']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Mini bar chart of session results
    if st.session_state.total >= 2:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["FAKE", "REAL"],
            y=[st.session_state.fake_n, st.session_state.real_n],
            marker_color=["#c0152a", "#34d399"],
            width=0.4,
        ))
        fig_bar.update_layout(
            height=160, margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#888", showlegend=False,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, tickfont=dict(color="#444")),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#1e293b;font-size:.78rem;letter-spacing:.04em;'>"
    "🛡️ TruthGuard AI &nbsp;·&nbsp; Intelligence Engine v2.5 &nbsp;·&nbsp; "
    "Powered by scikit-learn &amp; LLaMA 3.3 (Groq) &nbsp;·&nbsp; ISOT Dataset"
    "</p>",
    unsafe_allow_html=True,
)
