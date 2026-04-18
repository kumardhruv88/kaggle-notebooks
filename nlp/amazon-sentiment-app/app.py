import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Analyser",
    layout="wide",
)

# ── Download NLTK data (once) ──────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("omw-1.4",   quiet=True)

download_nltk()

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("lr_model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_models()

# ── Preprocessing ──────────────────────────────────────────────────────────────
stop_words  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>",      " ", text)   # HTML
    text = re.sub(r"[^a-z\s]",  " ", text)   # non-alpha
    text = " ".join(w for w in text.split() if w not in stop_words)
    text = " ".join(lemmatizer.lemmatize(w) for w in text.split())
    return text.strip()

# ── Prediction helper ──────────────────────────────────────────────────────────
def predict(text: str):
    cleaned = clean(text)
    vec     = vectorizer.transform([cleaned])
    label   = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return label, proba, cleaned


THEME_PALETTES = {
    "dark": {
        "fig_bg": "#0e1829",
        "ax_bg": "#0e1829",
        "label": "#95a7be",
        "title": "#d9e2ef",
        "ticks": "#afbdd1",
        "axis": "#3f5572",
        "legend_bg": "#13213a",
        "legend_text": "#e8edf5",
    },
    "light": {
        "fig_bg": "#f4f8ff",
        "ax_bg": "#f4f8ff",
        "label": "#44556f",
        "title": "#13243f",
        "ticks": "#536684",
        "axis": "#b9c8dc",
        "legend_bg": "#ebf2fc",
        "legend_text": "#1f304a",
    },
}

# ── Top-word contribution chart ────────────────────────────────────────────────
def contribution_chart(cleaned_text: str, theme_mode: str, n: int = 12):
    words      = cleaned_text.split()
    feat_names = np.array(vectorizer.get_feature_names_out())
    coefs      = model.coef_[0]
    theme = THEME_PALETTES.get(theme_mode, THEME_PALETTES["dark"])

    scores = {}
    for w in set(words):
        idxs = np.where(feat_names == w)[0]
        if len(idxs):
            scores[w] = coefs[idxs[0]]

    if not scores:
        return None

    sorted_items = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
    words_plot, vals_plot = zip(*sorted_items)

    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals_plot]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(theme["fig_bg"])
    ax.set_facecolor(theme["ax_bg"])

    bars = ax.barh(words_plot[::-1], vals_plot[::-1], color=colors[::-1],
                   edgecolor="none", height=0.6)

    ax.axvline(0, color=theme["axis"], linewidth=0.8)
    ax.set_xlabel("LR Coefficient", color=theme["label"], fontsize=9)
    ax.set_title("Word influence on prediction", color=theme["title"],
                 fontsize=10, pad=8)
    ax.tick_params(colors=theme["ticks"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    pos_patch = mpatches.Patch(color="#2ecc71", label="→ Positive")
    neg_patch = mpatches.Patch(color="#e74c3c", label="→ Negative")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8,
              facecolor=theme["legend_bg"], labelcolor=theme["legend_text"], loc="lower right")

    plt.tight_layout()
    return fig

# ── Theme selection ────────────────────────────────────────────────────────────
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

theme_toggle_col_left, theme_toggle_col_right = st.columns([0.82, 0.18])
with theme_toggle_col_right:
    light_mode = st.toggle("Light theme", value=st.session_state.theme_mode == "light")
st.session_state.theme_mode = "light" if light_mode else "dark"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
if st.session_state.theme_mode == "dark":
    root_vars = """
    --bg-main: #08101e;
    --bg-panel: #0e1829;
    --bg-soft: #13213a;
    --bg-elevated: #0f1d33;
    --text-main: #e8edf5;
    --text-muted: #9bacbf;
    --border: #243754;
    --accent: #20c997;
    --accent-strong: #16a085;
    --danger: #e15757;
    --shadow: 0 16px 42px rgba(2, 7, 18, 0.35);
    --panel-shadow: 0 10px 26px rgba(1, 7, 18, 0.26);
    --empty-grad-1: rgba(44, 84, 138, 0.18);
    --empty-grad-2: rgba(31, 191, 145, 0.14);
    --hero-bg: linear-gradient(125deg, #0f1c33 0%, #152b49 62%, #1d3557 100%);
    --card-pos-bg: linear-gradient(120deg, rgba(18, 88, 60, 0.75), rgba(17, 62, 47, 0.85));
    --card-neg-bg: linear-gradient(120deg, rgba(114, 45, 45, 0.72), rgba(78, 31, 31, 0.84));
    --btn-hover-bg: #1a2d4b;
    --btn-hover-border: #2f4e77;
    --btn-hover-text: #f5f8fc;
    --app-bg:
        radial-gradient(1200px 500px at 7% -10%, rgba(32, 201, 151, 0.14), transparent 55%),
        radial-gradient(1000px 520px at 92% -15%, rgba(80, 148, 255, 0.16), transparent 58%),
        #08101e;
    """
else:
    root_vars = """
    --bg-main: #eef4fc;
    --bg-panel: #f8fbff;
    --bg-soft: #e6eef9;
    --bg-elevated: #f1f6ff;
    --text-main: #0f223d;
    --text-muted: #5f7593;
    --border: #c9d7ea;
    --accent: #169b79;
    --accent-strong: #11745b;
    --danger: #d24c4c;
    --shadow: 0 16px 40px rgba(26, 63, 110, 0.12);
    --panel-shadow: 0 12px 30px rgba(36, 84, 140, 0.08);
    --empty-grad-1: rgba(40, 93, 168, 0.13);
    --empty-grad-2: rgba(22, 155, 121, 0.13);
    --hero-bg: linear-gradient(122deg, #edf4ff 0%, #d8e8ff 62%, #d1e4ff 100%);
    --card-pos-bg: linear-gradient(120deg, rgba(204, 244, 231, 0.92), rgba(184, 236, 221, 0.96));
    --card-neg-bg: linear-gradient(120deg, rgba(251, 221, 221, 0.94), rgba(245, 203, 203, 0.96));
    --btn-hover-bg: #dbe9fa;
    --btn-hover-border: #aec4e3;
    --btn-hover-text: #0f223d;
    --app-bg:
        radial-gradient(1200px 520px at 8% -12%, rgba(87, 140, 226, 0.18), transparent 58%),
        radial-gradient(1000px 520px at 90% -14%, rgba(57, 180, 145, 0.14), transparent 58%),
        #eef4fc;
    """

css_template = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

:root {
    __ROOT_VARS__
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: var(--text-main);
}

.stApp {
    background: var(--app-bg);
}

.block-container {
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.01em;
}

.hero {
    background: var(--hero-bg);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2.1rem 2rem 1.7rem;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow);
}

.hero h1 {
    font-size: clamp(1.9rem, 2.6vw, 2.7rem);
    font-weight: 700;
    line-height: 1.12;
    margin: 0 0 0.45rem 0;
}

.hero p {
    color: var(--text-muted);
    font-size: 0.98rem;
    margin: 0;
}

.panel {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1rem 1.1rem;
    box-shadow: var(--panel-shadow);
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.subtle-label {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-bottom: 0.45rem;
}

/* Result cards */
.result-card {
    border-radius: 14px;
    padding: 1.2rem 1.15rem 1rem;
    border: 1px solid transparent;
}

.card-pos {
    background: var(--card-pos-bg);
    border-color: rgba(32, 201, 151, 0.5);
}

.card-neg {
    background: var(--card-neg-bg);
    border-color: rgba(225, 87, 87, 0.48);
}

.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin: 0 0 0.15rem;
    text-transform: uppercase;
}

.result-conf {
    font-size: 0.96rem;
    color: rgba(232, 237, 245, 0.9);
    margin-bottom: 0.65rem;
}

/* Confidence bar */
.conf-bar-wrap {
    background: rgba(8, 16, 30, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}

/* Cleaned text box */
.cleaned-box {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    font-size: 0.84rem;
    color: #b7c4d4;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    word-break: break-word;
}

/* Stat chips */
.chip {
    display: inline-block;
    background: var(--bg-soft);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.34rem 0.78rem;
    font-size: 0.82rem;
    color: #bdd1e8;
    margin: 0.2rem 0.26rem 0 0;
}

/* Textarea */
textarea {
    background-color: var(--bg-panel) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Manrope', sans-serif !important;
    line-height: 1.45 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted);
}

[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--bg-panel);
}

.stButton > button {
    width: 100%;
    background: var(--bg-soft);
    border: 1px solid var(--border);
    color: #c5d4e6;
    border-radius: 10px;
    font-size: 0.89rem;
    font-weight: 500;
    padding: 0.42rem 0.7rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: var(--btn-hover-bg);
    border-color: var(--btn-hover-border);
    color: var(--btn-hover-text);
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-strong));
    border: 0;
    color: #06231b;
    font-weight: 700;
}

div.stButton > button[kind="primary"]:hover {
    filter: brightness(1.03);
}

hr {
    border-color: rgba(62, 86, 115, 0.45);
}

.empty-state {
    background:
        radial-gradient(420px 170px at 15% 10%, var(--empty-grad-1), transparent 70%),
        radial-gradient(350px 170px at 85% 20%, var(--empty-grad-2), transparent 75%),
        var(--bg-panel);
    border: 1px dashed var(--border);
    border-radius: 14px;
    padding: 2.4rem 1.4rem;
    text-align: center;
}

.empty-state .line-1 {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text-main);
    font-size: 1.05rem;
    margin-bottom: 0.35rem;
}

.empty-state .line-2 {
    color: var(--text-muted);
    font-size: 0.91rem;
    max-width: 360px;
    margin: 0 auto;
    line-height: 1.5;
}

[data-testid="stToggle"] {
    margin-top: 0.35rem;
}

@media (max-width: 900px) {
    .hero {
        padding: 1.5rem 1.2rem 1.2rem;
    }

    .hero p {
        font-size: 0.9rem;
    }

    .panel {
        padding: 0.85rem 0.8rem 0.9rem;
    }
}
</style>
"""

st.markdown(css_template.replace("__ROOT_VARS__", root_vars), unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Amazon Review Analyser</h1>
  <p>Powered by Logistic Regression + TF-IDF · Trained on 568K Amazon Fine Food Reviews · 90.5% accuracy</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1.1, 0.9], gap="large")

# ── Example reviews ────────────────────────────────────────────────────────────
EXAMPLES = {
    "Glowing review":      "Absolutely amazing product! Best coffee I've ever tasted. Highly recommend to everyone.",
    "Angry review":        "Terrible quality. Completely disappointed. Waste of money, never buying again.",
    "Mixed review":        "The dog loved it but the packaging was horrible and leaking everywhere.",
    "Neutral review":      "It was okay, nothing special but not bad either. Pretty average product.",
    "Delivery complaint":  "Product itself is fine but arrived damaged and the box was completely crushed.",
}

with col_input:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Enter a Review</div>', unsafe_allow_html=True)

    # Example buttons
    st.markdown('<div class="subtle-label">Quick examples</div>', unsafe_allow_html=True)
    btn_cols = st.columns(3)
    selected_example = None
    for i, (label, review) in enumerate(EXAMPLES.items()):
        if btn_cols[i % 3].button(label, key=f"ex_{i}"):
            selected_example = review

    default_text = selected_example if selected_example else ""
    review_input = st.text_area(
        label="review_area",
        value=default_text,
        placeholder="Paste or type any Amazon product review here.",
        height=160,
        label_visibility="collapsed",
    )

    analyse_btn = st.button("Analyse Sentiment", type="primary", use_container_width=True)

    # Stats about the model
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="subtle-label">Model info</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="chip">568K reviews trained</span>
    <span class="chip">90.5% accuracy</span>
    <span class="chip">10K TF-IDF features</span>
    <span class="chip">Bigrams included</span>
    <span class="chip">Balanced classes</span>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Results ────────────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

    if analyse_btn and review_input.strip():
        with st.spinner("Analysing review..."):
            label, proba, cleaned = predict(review_input)

        pos_conf = proba[1] * 100
        neg_conf = proba[0] * 100
        is_pos   = label == 1

        # Result card
        if is_pos:
            st.markdown(f"""
                        <div class="result-card card-pos">
              <div class="result-label" style="color:#2ecc71">POSITIVE</div>
              <div class="result-conf">Confidence: {pos_conf:.1f}%</div>
              <div class="conf-bar-wrap" style="margin-top:.8rem">
                <div class="conf-bar-fill" style="width:{pos_conf}%;background:#2ecc71"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                        <div class="result-card card-neg">
              <div class="result-label" style="color:#e74c3c">NEGATIVE</div>
              <div class="result-conf">Confidence: {neg_conf:.1f}%</div>
              <div class="conf-bar-wrap" style="margin-top:.8rem">
                <div class="conf-bar-fill" style="width:{neg_conf}%;background:#e74c3c"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Positive", f"{pos_conf:.1f}%")
        c2.metric("Negative", f"{neg_conf:.1f}%")

        # Word influence chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig = contribution_chart(cleaned, st.session_state.theme_mode)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No known words found in vocabulary.")

        # Cleaned text
        with st.expander("See preprocessed text"):
            st.markdown(f'<div class="cleaned-box">{cleaned}</div>',
                        unsafe_allow_html=True)

    elif analyse_btn and not review_input.strip():
        st.warning("Please enter a review first!")
    else:
        st.markdown("""
                <div class="empty-state">
                    <div class="line-1">Ready for analysis</div>
                    <div class="line-2">
                        Enter a review on the left and click Analyse Sentiment to view sentiment,
                        confidence scores, and keyword influence.
                    </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#445;font-size:.8rem;padding:.5rem 0">
    Built with Logistic Regression + TF-IDF · Amazon Fine Food Reviews Dataset
</div>
""", unsafe_allow_html=True)