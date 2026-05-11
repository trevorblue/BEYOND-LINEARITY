"""
Beyond Linearity v2 — California Housing Analysis
MSc Data Science & Analytics · Premium Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(
    page_title="Beyond Linearity v2",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stMarkdown, p, li, span {
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #cbd5e1;
}

#MainMenu, footer, header { visibility: hidden; }

.stApp { background: #070b14 !important; }

.block-container {
    padding-top: 1.8rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1440px !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1220 0%, #0a1028 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.12) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}
[data-testid="stSidebar"] .stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.87rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    padding: 0.5rem 0.9rem !important;
    border-radius: 8px !important;
    transition: background 0.2s, color 0.2s !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,179,237,0.08) !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] { display: flex !important; }
[data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ div,
[data-testid="stSidebar"] .stRadio [data-checked="true"] ~ div {
    color: #63b3ed !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.8rem !important;
    color: #475569 !important;
    line-height: 1.6 !important;
}

/* ── Typography ─────────────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f1f5f9 !important;
    font-weight: 700 !important;
}
h1 { font-size: 2.1rem !important; letter-spacing: -0.5px !important; line-height: 1.2 !important; }
h2 { font-size: 1.4rem !important; letter-spacing: -0.3px !important; border: none !important; }
h3 { font-size: 1.05rem !important; color: #94a3b8 !important; }
hr { border-color: rgba(99,179,237,0.1) !important; margin: 1.5rem 0 !important; }
strong { color: #e2e8f0 !important; }

/* ── Sidebar toggle button ──────────────────────────────────────────────────── */
[data-testid="collapsedControl"] {
    background: linear-gradient(135deg, #63b3ed, #a78bfa) !important;
    border-radius: 0 10px 10px 0 !important;
    width: 28px !important;
    top: 50% !important;
    color: #fff !important;
    box-shadow: 4px 0 16px rgba(99,179,237,0.4) !important;
}
[data-testid="collapsedControl"] svg { fill: #fff !important; stroke: #fff !important; }
[data-testid="stSidebarCollapseButton"] button {
    color: #63b3ed !important;
    background: rgba(99,179,237,0.1) !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0c1220; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2d4263; }

/* ── Animations ─────────────────────────────────────────────────────────────── */
@keyframes gradientFlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 10px rgba(99,179,237,0.2); }
    50%       { box-shadow: 0 0 28px rgba(99,179,237,0.45); }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ── Reusable Components ────────────────────────────────────────────────────── */

.gradient-text {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed 0%, #a78bfa 40%, #34d399 70%, #63b3ed 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientFlow 7s ease infinite;
}

.page-hero {
    background: linear-gradient(135deg,
        rgba(99,179,237,0.07) 0%,
        rgba(167,139,250,0.05) 50%,
        rgba(52,211,153,0.04) 100%);
    border: 1px solid rgba(99,179,237,0.14);
    border-radius: 18px;
    padding: 2rem 2.5rem 1.8rem;
    margin-bottom: 1.8rem;
    animation: fadeUp 0.55s ease;
}
.hero-subtitle {
    font-size: 0.92rem;
    color: #64748b;
    margin-top: 0.4rem;
    line-height: 1.6;
    max-width: 780px;
}

.metric-card {
    background: linear-gradient(135deg, rgba(99,179,237,0.09), rgba(167,139,250,0.06));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 14px;
    padding: 1.3rem 1rem;
    text-align: center;
    animation: fadeUp 0.5s ease;
    transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 36px rgba(99,179,237,0.16);
    border-color: rgba(99,179,237,0.4);
}
.metric-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #475569;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #63b3ed;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.35rem;
}

.glass-card {
    background: rgba(12, 18, 32, 0.85);
    border: 1px solid rgba(99,179,237,0.13);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin: 0.7rem 0;
    backdrop-filter: blur(8px);
    transition: border-color 0.3s, box-shadow 0.3s;
    animation: fadeUp 0.5s ease;
}
.glass-card:hover {
    border-color: rgba(99,179,237,0.28);
    box-shadow: 0 4px 28px rgba(99,179,237,0.08);
}

.callout {
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin: 0.75rem 0;
    border-left: 3px solid;
    font-size: 0.87rem;
    line-height: 1.65;
}
.callout-blue   { background: rgba(99,179,237,0.07);  border-color: #63b3ed; color: #bfdbfe; }
.callout-green  { background: rgba(52,211,153,0.07);  border-color: #34d399; color: #a7f3d0; }
.callout-amber  { background: rgba(251,191,36,0.07);  border-color: #fbbf24; color: #fef3c7; }
.callout-purple { background: rgba(167,139,250,0.07); border-color: #a78bfa; color: #ede9fe; }
.callout-red    { background: rgba(248,113,113,0.07); border-color: #f87171; color: #fecaca; }

.badge {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.2px;
    margin: 2px 3px;
}
.badge-blue   { background: rgba(99,179,237,0.13); color: #63b3ed;  border: 1px solid rgba(99,179,237,0.28); }
.badge-purple { background: rgba(167,139,250,0.13); color: #a78bfa; border: 1px solid rgba(167,139,250,0.28); }
.badge-green  { background: rgba(52,211,153,0.13); color: #34d399;  border: 1px solid rgba(52,211,153,0.28); }
.badge-amber  { background: rgba(251,191,36,0.13); color: #fbbf24;  border: 1px solid rgba(251,191,36,0.28); }
.badge-red    { background: rgba(248,113,113,0.13); color: #f87171; border: 1px solid rgba(248,113,113,0.28); }

.section-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1rem;
    animation: slideIn 0.4s ease;
}
.section-num {
    background: linear-gradient(135deg, #63b3ed, #a78bfa);
    color: #fff;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.8rem;
    min-width: 30px;
    height: 30px;
    border-radius: 7px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.section-title-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
}
.section-desc {
    font-size: 0.82rem;
    color: #475569;
    margin-top: 2px;
}

.big-quote {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    color: #94a3b8;
    border-left: 3px solid #63b3ed;
    padding: 1rem 1.5rem;
    background: rgba(99,179,237,0.05);
    border-radius: 0 10px 10px 0;
    margin: 1.2rem 0;
    line-height: 1.75;
}

.winner-banner {
    background: linear-gradient(135deg, rgba(52,211,153,0.1), rgba(99,179,237,0.08));
    border: 1px solid rgba(52,211,153,0.35);
    border-radius: 16px;
    padding: 1.4rem 2rem;
    animation: pulseGlow 3.5s ease-in-out infinite;
    margin: 1rem 0 1.5rem;
}
.winner-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #34d399;
    margin-bottom: 0.4rem;
}
.winner-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
}
.winner-stats {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.4rem;
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.winner-stat-item { display: flex; flex-direction: column; }
.winner-stat-val  { font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600; color: #34d399; }
.winner-stat-lbl  { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 0.8px; }

.rank-medal {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    font-weight: 700;
    font-size: 0.78rem;
}
.rank-1 { background: linear-gradient(135deg, #f59e0b, #fbbf24); color: #000; }
.rank-2 { background: linear-gradient(135deg, #94a3b8, #cbd5e1); color: #000; }
.rank-3 { background: linear-gradient(135deg, #b45309, #d97706); color: #fff; }
.rank-n { background: rgba(99,179,237,0.12); color: #63b3ed; }

.prediction-box {
    background: linear-gradient(135deg, rgba(99,179,237,0.13), rgba(52,211,153,0.09));
    border: 1px solid rgba(99,179,237,0.38);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    animation: pulseGlow 3.5s ease-in-out infinite;
}
.pred-amount {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.pred-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    color: #475569;
    margin-bottom: 0.5rem;
}

.improvement-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 24px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: #94a3b8;
    margin: 3px;
    transition: all 0.2s;
}
.improvement-pill:hover {
    border-color: rgba(99,179,237,0.35);
    color: #e2e8f0;
    background: rgba(99,179,237,0.08);
}

.code-block {
    background: rgba(7,11,20,0.9);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    color: #34d399;
    white-space: pre;
    overflow-x: auto;
}

.fig-wrap {
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 12px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.fig-caption {
    text-align: center;
    font-size: 0.75rem;
    color: #475569;
    font-style: italic;
    padding: 0.5rem;
    background: rgba(12,18,32,0.6);
}

/* Widget overrides */
.stSlider > div > div > div { background: rgba(99,179,237,0.15) !important; }
.stNumberInput input {
    background: rgba(12,18,32,0.9) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(99,179,237,0.12) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
.stAlert { border-radius: 10px !important; }

/* ── Tab navigation ─────────────────────────────────────────────────────────── */
.stTabs [data-testid="stTabsNavContainer"] {
    background: rgba(12,18,32,0.9) !important;
    border-bottom: 1px solid rgba(99,179,237,0.15) !important;
    border-radius: 14px 14px 0 0 !important;
    padding: 0 0.5rem !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    padding: 0.75rem 1.2rem !important;
    border-radius: 10px 10px 0 0 !important;
    transition: color 0.2s, background 0.2s !important;
    border: none !important;
    letter-spacing: 0.2px !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #94a3b8 !important;
    background: rgba(99,179,237,0.06) !important;
}
.stTabs [aria-selected="true"] {
    color: #63b3ed !important;
    background: rgba(99,179,237,0.1) !important;
    border-bottom: 2px solid #63b3ed !important;
}
.stTabs [data-testid="stTabContent"] {
    padding-top: 1.5rem !important;
}

/* Hide sidebar entirely — nav is via tabs */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artefacts…")
def load_artifacts():
    m = BASE / "models"
    required = ["xgb_model.pkl", "sigma2.json", "results_df.csv",
                "features.json", "diagnostics.json", "dm_results.json"]
    missing = [f for f in required if not (m / f).exists()]
    if missing:
        return None, missing
    model       = joblib.load(m / "xgb_model.pkl")
    sigma2      = json.loads((m / "sigma2.json").read_text())
    results_df  = pd.read_csv(m / "results_df.csv")
    features    = json.loads((m / "features.json").read_text())
    diagnostics = json.loads((m / "diagnostics.json").read_text())
    dm_results  = json.loads((m / "dm_results.json").read_text())
    return (model, sigma2, results_df, features, diagnostics, dm_results), []

artifacts, missing_files = load_artifacts()

if missing_files:
    st.error(
        f"**Missing artefacts:** `{', '.join(missing_files)}`\n\n"
        "Run **Section 17** in `Beyond_Linearity_v2.ipynb`, then refresh."
    )
    st.stop()

model, sigma2, results_df, features, diagnostics, dm_results = artifacts
features_tree   = features["features_tree"]
features_linear = features["features_linear"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def show_fig(name, caption=""):
    p = BASE / name
    if p.exists():
        st.markdown('<div class="fig-wrap">', unsafe_allow_html=True)
        st.image(str(p), use_container_width=True)
        if caption:
            st.markdown(f'<div class="fig-caption">{caption}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def chart_style(fig, title="", height=400):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,18,32,0.6)",
        font=dict(family="Inter, sans-serif", color="#64748b", size=12),
        title=dict(
            text=title,
            font=dict(family="Space Grotesk, sans-serif", size=14, color="#f1f5f9"),
            pad=dict(b=10),
        ),
        height=height,
        margin=dict(l=10, r=60, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    )
    fig.update_xaxes(
        gridcolor="rgba(99,179,237,0.07)",
        zerolinecolor="rgba(99,179,237,0.15)",
        tickfont=dict(color="#475569"),
        title_font=dict(color="#64748b"),
    )
    fig.update_yaxes(
        gridcolor="rgba(99,179,237,0.07)",
        zerolinecolor="rgba(99,179,237,0.15)",
        tickfont=dict(color="#475569"),
        title_font=dict(color="#64748b"),
    )
    return fig


# Colour palette
C_BLUE   = "#63b3ed"
C_PURPLE = "#a78bfa"
C_GREEN  = "#34d399"
C_AMBER  = "#fbbf24"
C_RED    = "#f87171"

rdf      = results_df.sort_values("rmse_log").reset_index(drop=True)
best_row = rdf.iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# TOP NAV — App header + tabs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:0.8rem 0 1.2rem;border-bottom:1px solid rgba(99,179,237,0.12);
            margin-bottom:1rem;">
  <div>
    <span style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;
                 color:#f1f5f9;">Beyond Linearity</span>
    <span style="font-family:'Space Grotesk',sans-serif;font-size:0.72rem;font-weight:600;
                 color:#63b3ed;letter-spacing:1px;text-transform:uppercase;
                 margin-left:0.5rem;">v2.0</span>
  </div>
  <div style="display:flex;gap:1.5rem;font-size:0.78rem;color:#475569;">
    <span>Best: <strong style="color:#63b3ed;">{best_row['name']}</strong></span>
    <span>R² <strong style="color:#34d399;">{best_row['r2']:.4f}</strong></span>
    <span>RMSE <strong style="color:#a78bfa;">${best_row['rmse_dollar']:,.0f}</strong></span>
    <span style="color:#334155;">20,640 block groups · CA 1990</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Model Leaderboard",
    "Statistical Diagnostics",
    "Spatial Analysis",
    "Live Prediction",
])


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 · OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("""
    <div class="page-hero">
      <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:1.5px;color:#475569;margin-bottom:0.5rem;">
        MSc Data Science &amp; Analytics · California Housing
      </div>
      <div class="gradient-text" style="font-size:2.3rem;font-weight:700;
                                        letter-spacing:-0.5px;line-height:1.15;">
        Beyond Linearity v2
      </div>
      <div class="hero-subtitle">
        14 methodological improvements over the naïve implementation —
        rigorous regression analysis that produces numbers you can
        <em>trust</em>, interpret in <em>practical units</em>, and
        <em>defend to a statistician</em>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key stats ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Best Test R²</div>
          <div class="metric-value">{best_row['r2']:.4f}</div>
          <div class="metric-sub">XGBoost · held-out 20%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Models Compared</div>
          <div class="metric-value">9</div>
          <div class="metric-sub">OLS → XGBoost pipeline</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Methodological Fixes</div>
          <div class="metric-value">14</div>
          <div class="metric-sub">v1 → v2 improvements</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Dataset Size</div>
          <div class="metric-value">20,640</div>
          <div class="metric-sub">block groups · 1990 Census</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Central thesis ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="big-quote">
      "v1 produced numbers. v2 produces numbers you can <strong style='color:#63b3ed'>trust</strong>,
      interpret in <strong style='color:#34d399'>practical units</strong>, and
      <strong style='color:#a78bfa'>defend to a statistician</strong>."
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "Most ML tutorials report a test R² and call it done. This analysis goes further: "
        "running the statistical tests that expose *why* a model fails, correcting the "
        "back-transformation bias that inflates every dollar-scale prediction, and honestly "
        "documenting the information ceiling that no methodology can breach."
    )

    st.markdown("---")

    # ── 14 improvement badges ──────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">14</span>
      <div>
        <div class="section-title-text">All Methodological Improvements</div>
        <div class="section-desc">Every addition v2 makes over the naïve v1 implementation</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin:0.5rem 0 1rem;">
      <span class="badge badge-blue">VIF Multicollinearity Audit</span>
      <span class="badge badge-blue">Breusch-Pagan Heteroscedasticity Test</span>
      <span class="badge badge-blue">Diebold-Mariano Significance Test</span>
      <span class="badge badge-blue">Moran's I Spatial Autocorrelation</span>
      <span class="badge badge-blue">Train vs Test Overfitting Gap</span>
      <span class="badge badge-purple">Jensen's Inequality Bias Correction</span>
      <span class="badge badge-purple">σ² per-model Back-Transformation</span>
      <span class="badge badge-green">Winsorisation at 99th Percentile</span>
      <span class="badge badge-green">RidgeCV Cross-Validated α</span>
      <span class="badge badge-green">LassoCV Cross-Validated α</span>
      <span class="badge badge-amber">rooms_per_household Feature</span>
      <span class="badge badge-amber">bedrooms_per_room Feature</span>
      <span class="badge badge-amber">population_per_household Feature</span>
      <span class="badge badge-red">Percentile-Based Region Classification Fix</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 3 categories ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">3</span>
      <div>
        <div class="section-title-text">Three Categories of Improvement</div>
        <div class="section-desc">How each fix changes (or doesn't change) R²</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-card" style="border-color:rgba(99,179,237,0.2);">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:0.8rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.8px;color:#63b3ed;
                      margin-bottom:0.8rem;padding-bottom:0.6rem;
                      border-bottom:1px solid rgba(99,179,237,0.12);">
            Category 1 · Diagnostic Additions
          </div>
          <ul style="padding-left:1.1rem;margin:0;font-size:0.85rem;line-height:2;">
            <li><strong>VIF</strong> — multicollinearity audit</li>
            <li><strong>Breusch-Pagan</strong> — heteroscedasticity</li>
            <li><strong>Diebold-Mariano</strong> — significance vs OLS</li>
            <li><strong>Moran's I</strong> — spatial residual clustering</li>
            <li><strong>Overfitting gap</strong> — train vs test R² exposed</li>
          </ul>
          <div style="margin-top:1rem;font-size:0.75rem;color:#475569;
                      border-top:1px solid rgba(99,179,237,0.08);padding-top:0.6rem;">
            Effect on R²: <span style="color:#f87171;font-weight:600;">none</span> —
            exposes failures v1 silently ignored
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card" style="border-color:rgba(167,139,250,0.2);">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:0.8rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.8px;color:#a78bfa;
                      margin-bottom:0.8rem;padding-bottom:0.6rem;
                      border-bottom:1px solid rgba(167,139,250,0.12);">
            Category 2 · Back-Transformation
          </div>
          <ul style="padding-left:1.1rem;margin:0;font-size:0.85rem;line-height:2;">
            <li><strong>Jensen's correction</strong> [Duan, 1983]</li>
            <li>Corrected: <code style="color:#34d399;font-size:0.8rem;">exp(ŷ + σ²/2)</code></li>
            <li>Naïve: <code style="color:#f87171;font-size:0.8rem;">exp(ŷ)</code></li>
            <li>Removes systematic underestimation</li>
            <li>σ² estimated per-model from training residuals</li>
          </ul>
          <div style="margin-top:1rem;font-size:0.75rem;color:#475569;
                      border-top:1px solid rgba(167,139,250,0.08);padding-top:0.6rem;">
            Effect on R²: <span style="color:#fbbf24;font-weight:600;">none</span> —
            but dollar RMSE becomes honest
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass-card" style="border-color:rgba(52,211,153,0.2);">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:0.8rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.8px;color:#34d399;
                      margin-bottom:0.8rem;padding-bottom:0.6rem;
                      border-bottom:1px solid rgba(52,211,153,0.12);">
            Category 3 · Data &amp; Hyperparameter Fixes
          </div>
          <ul style="padding-left:1.1rem;margin:0;font-size:0.85rem;line-height:2;">
            <li><strong>Winsorisation</strong> — pop/hh clipped at p99 (max was 1,243!)</li>
            <li><strong>RidgeCV</strong> — cross-validated α (was hardcoded α=10)</li>
            <li><strong>LassoCV</strong> — cross-validated α (was hardcoded α=0.01)</li>
            <li><strong>Engineered features</strong> — 3 interaction ratios added</li>
            <li><strong>Region fix</strong> — percentile thresholds, not hardcoded</li>
          </ul>
          <div style="margin-top:1rem;font-size:0.75rem;color:#475569;
                      border-top:1px solid rgba(52,211,153,0.08);padding-top:0.6rem;">
            Effect on R²: <span style="color:#34d399;font-weight:600;">direct</span> —
            for Ridge, Lasso, and tree models
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── v1 vs v2 table ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">9</span>
      <div>
        <div class="section-title-text">v1 vs v2 — Full Model Comparison</div>
        <div class="section-desc">Why the R² values are similar — and why that's the point</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cmp = pd.DataFrame({
        "Model":          ["OLS", "Ridge", "Lasso", "Spline", "GAM",
                           "Neural Net", "Random Forest", "Gradient Boosting", "XGBoost"],
        "v1 R²":          [0.6330, 0.6330, 0.6185, 0.7175, 0.7591,
                           0.7452, 0.8228, 0.8368, 0.8505],
        "v2 R² (approx)": ["≈0.633", "≈0.633", "≈0.620", "≈0.718", "≈0.760",
                           "≈0.746", "≈0.822", "≈0.837", "≈0.851"],
        "Primary Change":  [
            "Diagnostics added — no R² shift",
            "RidgeCV selected α ≈ hardcoded value → negligible shift",
            "LassoCV tightened regularisation slightly",
            "Diagnostics only",
            "Diagnostics only",
            "Diagnostics only",
            "Overfitting gap exposed (train 0.97 vs test 0.82)",
            "Diagnostics only",
            "Winner unchanged — now statistically validated by DM test",
        ],
    })
    st.dataframe(cmp, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="callout callout-blue">
      <strong>Why are R² values so similar?</strong> The remaining 15% unexplained variance is an
      <em>information ceiling</em> — school quality, crime, zoning, and employment proximity were
      never in the 1990 Census. No methodology can recover information that was never collected.
      The value of v2 is a <em>defensible</em> number with honest uncertainty, not a higher one.
      See the <strong>Spatial Analysis</strong> page for a quantified breakdown.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        show_fig("v2_fig1_target_distribution.png",
                 "Fig 1 · Target distribution — log-normal shape; 4.7% censored at $500,001")
    with col2:
        show_fig("v2_fig2_ceiling_effect.png",
                 "Fig 2 · Ceiling effect — censoring artefact visible at top of price range")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 · MODEL LEADERBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="page-hero">
      <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:1.5px;color:#475569;margin-bottom:0.5rem;">Model Leaderboard</div>
      <div class="gradient-text" style="font-size:2rem;font-weight:700;letter-spacing:-0.4px;">
        9 Models Ranked by Performance
      </div>
      <div class="hero-subtitle">
        All metrics on held-out 20% test set ·
        Dollar RMSE uses Jensen's inequality bias correction [Duan, 1983] ·
        Significance validated by Diebold-Mariano test [Harvey et al., 1997]
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Winner banner ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="winner-banner">
      <div class="winner-label">🏆 Champion Model</div>
      <div class="winner-name">{best_row['name']}</div>
      <div class="winner-stats">
        <div class="winner-stat-item">
          <span class="winner-stat-val">{best_row['r2']:.4f}</span>
          <span class="winner-stat-lbl">Test R²</span>
        </div>
        <div class="winner-stat-item">
          <span class="winner-stat-val">{best_row['rmse_log']:.4f}</span>
          <span class="winner-stat-lbl">Log RMSE</span>
        </div>
        <div class="winner-stat-item">
          <span class="winner-stat-val">${best_row['rmse_dollar']:,.0f}</span>
          <span class="winner-stat-lbl">Dollar RMSE (Jensen-corrected)</span>
        </div>
        <div class="winner-stat-item">
          <span class="winner-stat-val">${best_row['mae_dollar']:,.0f}</span>
          <span class="winner-stat-lbl">Dollar MAE</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    bar_colors = [C_GREEN if n == best_row["name"] else
                  C_BLUE  if i < 3 else "#2d4263"
                  for i, n in enumerate(rdf["name"])]

    with col1:
        fig_r2 = go.Figure(go.Bar(
            x=rdf["r2"], y=rdf["name"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.4f}" for v in rdf["r2"]],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
            hovertemplate="%{y}: R² = %{x:.4f}<extra></extra>",
        ))
        fig_r2.update_layout(xaxis=dict(range=[0.55, 0.93], title="R²"))
        chart_style(fig_r2, title="Test R²  (higher is better)", height=380)
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        fig_rmse = go.Figure(go.Bar(
            x=rdf["rmse_dollar"] / 1000, y=rdf["name"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"${v:,.1f}k" for v in rdf["rmse_dollar"] / 1000],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
            hovertemplate="%{y}: $%{x:.1f}k RMSE<extra></extra>",
        ))
        fig_rmse.update_layout(xaxis=dict(title="RMSE ($000s)"))
        chart_style(fig_rmse, title="Dollar RMSE — Jensen-corrected  (lower is better)", height=380)
        st.plotly_chart(fig_rmse, use_container_width=True)

    # ── Full results table ──────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">↓</span>
      <div>
        <div class="section-title-text">Full Results Table</div>
        <div class="section-desc">All 9 models sorted by log-scale RMSE</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    disp = rdf.copy()
    ranks = {0: "🥇", 1: "🥈", 2: "🥉"}
    disp.insert(0, "Rank", [ranks.get(i, f"#{i+1}") for i in range(len(disp))])
    disp["rmse_dollar"] = disp["rmse_dollar"].apply(lambda x: f"${x:,.0f}")
    disp["mae_dollar"]  = disp["mae_dollar"].apply(lambda x: f"${x:,.0f}")
    disp["r2"]          = disp["r2"].round(4)
    disp["rmse_log"]    = disp["rmse_log"].round(4)
    disp = disp.rename(columns={
        "name": "Model", "rmse_log": "Log RMSE", "r2": "Test R²",
        "rmse_dollar": "RMSE ($)", "mae_dollar": "MAE ($)",
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── DM Test ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">DM</span>
      <div>
        <div class="section-title-text">Diebold-Mariano Test — Statistical Significance vs OLS</div>
        <div class="section-desc">
          p &lt; 0.05 = the improvement is real, not sampling noise · [Harvey, Leybourne &amp; Newbold, 1997]
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-purple">
      A model with higher R² that is <em>not</em> DM-significant means the improvement may be
      sampling noise. This test is what separates a rigorous analysis from a leaderboard snapshot.
      v1 did not run this test at all.
    </div>
    """, unsafe_allow_html=True)

    dm_rows = []
    for mname, res in dm_results.items():
        sig = res["p_value"] < 0.05
        dm_rows.append({
            "Model":        mname,
            "DM Statistic": round(res["dm_stat"], 3),
            "p-value":      round(res["p_value"], 4),
            "Significant":  "✅ Yes — better than OLS" if sig else "⚠️ Not distinguishable from OLS",
        })
    dm_df = pd.DataFrame(dm_rows).sort_values("p-value").reset_index(drop=True)
    st.dataframe(dm_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        show_fig("v2_fig8_predicted_actual.png",
                 "Fig 8 · Predicted vs Actual — OLS (left) vs best model (right)")
    with col2:
        show_fig("v2_fig9_cross_validation.png",
                 "Fig 9 · 5-fold cross-validation — all models (mean ± std)")

    st.markdown("---")
    show_fig("v2_fig7_model_comparison.png",
             "Fig 7 · Full model comparison — log-scale RMSE with 95% confidence intervals")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 · STATISTICAL DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown("""
    <div class="page-hero">
      <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:1.5px;color:#475569;margin-bottom:0.5rem;">
        Statistical Diagnostics
      </div>
      <div class="gradient-text" style="font-size:2rem;font-weight:700;letter-spacing:-0.4px;">
        The Tests v1 Skipped
      </div>
      <div class="hero-subtitle">
        Each test below exposes a different failure mode that a naïve implementation
        silently ignores — these are the results that make the analysis defensible.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 1. VIF ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">1</span>
      <div>
        <div class="section-title-text">Variance Inflation Factor — Multicollinearity Audit</div>
        <div class="section-desc">Measures how much each predictor's variance is inflated by collinearity</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-blue">
      <strong>Why this matters:</strong> VIF explains <em>why Ridge R² ≈ OLS R²</em>.
      If multicollinearity were the bottleneck, Ridge regularisation would have helped — it didn't.
      Therefore the bottleneck is <strong>nonlinearity</strong>, not collinear predictors.
      VIF &gt; 10 = severe · VIF &gt; 5 = investigate · VIF ≤ 5 = acceptable.
    </div>
    """, unsafe_allow_html=True)

    vif_df = pd.DataFrame(diagnostics["vif"])
    vif_df["VIF"] = vif_df["VIF"].round(2)

    def vif_flag(v):
        if v > 10: return "🔴  Severe"
        if v > 5:  return "🟡  Moderate"
        return "🟢  OK"

    vif_df["Status"] = vif_df["VIF"].apply(vif_flag)
    vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.dataframe(vif_df, use_container_width=True, hide_index=True)
    with col2:
        vif_colors = [C_RED if v > 10 else C_AMBER if v > 5 else C_GREEN
                      for v in vif_df["VIF"]]
        fig_vif = go.Figure(go.Bar(
            x=vif_df["VIF"], y=vif_df["Feature"],
            orientation="h",
            marker=dict(color=vif_colors, line=dict(width=0)),
            text=[f"{v:.1f}" for v in vif_df["VIF"]],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        fig_vif.add_vline(x=5,  line_dash="dot", line_color=C_AMBER, line_width=1.5,
                          annotation_text="VIF = 5", annotation_font_color=C_AMBER,
                          annotation_font_size=11)
        fig_vif.add_vline(x=10, line_dash="dot", line_color=C_RED, line_width=1.5,
                          annotation_text="VIF = 10", annotation_font_color=C_RED,
                          annotation_font_size=11)
        chart_style(fig_vif, title="VIF by Feature", height=370)
        st.plotly_chart(fig_vif, use_container_width=True)

    st.markdown("---")

    # ── 2. Feature Engineering ──────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">2</span>
      <div>
        <div class="section-title-text">Feature Engineering — 3 Interaction Ratios</div>
        <div class="section-desc">Constructed to capture housing density and bedroom composition</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    fe_col1, fe_col2, fe_col3 = st.columns(3)
    with fe_col1:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:0.78rem;font-weight:700;
                      color:#63b3ed;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.6rem;">
            rooms_per_household
          </div>
          <div class="code-block" style="text-align:left;font-size:0.79rem;">
total_rooms ÷ households</div>
          <div style="font-size:0.8rem;color:#64748b;margin-top:0.6rem;">
            Captures crowding; a studio in SF vs a ranch in Fresno have very different values.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with fe_col2:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:0.78rem;font-weight:700;
                      color:#a78bfa;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.6rem;">
            bedrooms_per_room
          </div>
          <div class="code-block" style="text-align:left;font-size:0.79rem;">
total_bedrooms ÷ total_rooms</div>
          <div style="font-size:0.8rem;color:#64748b;margin-top:0.6rem;">
            Captures unit type mix. High ratio = bedroom-heavy blocks (dorms, SROs).
          </div>
        </div>
        """, unsafe_allow_html=True)
    with fe_col3:
        p99_val = diagnostics.get("pop_per_hh_p99", 10.0)
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:0.78rem;font-weight:700;
                      color:#34d399;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.6rem;">
            population_per_household
          </div>
          <div class="code-block" style="text-align:left;font-size:0.79rem;">
min(population ÷ households,
    p99 = {p99_val:.2f})</div>
          <div style="font-size:0.8rem;color:#64748b;margin-top:0.6rem;">
            Winsorised — raw max was 1,243 (380 std devs above mean of 3.07).
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 3. Breusch-Pagan ────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">3</span>
      <div>
        <div class="section-title-text">Breusch-Pagan Test — Heteroscedasticity</div>
        <div class="section-desc">Tests whether OLS Gauss-Markov assumption 4 holds</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    bp_stat = diagnostics["bp_stat"]
    bp_pval = diagnostics["bp_pval"]

    bpc1, bpc2, bpc3 = st.columns(3)
    with bpc1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">BP Statistic</div>
          <div class="metric-value" style="font-size:1.7rem;">{bp_stat:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with bpc2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">p-value</div>
          <div class="metric-value" style="font-size:1.7rem;color:#f87171;">{bp_pval:.2e}</div>
        </div>""", unsafe_allow_html=True)
    with bpc3:
        verdict_color = "#f87171" if bp_pval < 0.05 else "#34d399"
        verdict_text  = "Heteroscedastic" if bp_pval < 0.05 else "Homoscedastic"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Verdict</div>
          <div class="metric-value" style="font-size:1.4rem;color:{verdict_color};">
            {verdict_text}
          </div>
        </div>""", unsafe_allow_html=True)

    if bp_pval < 0.05:
        st.markdown(f"""
        <div class="callout callout-amber" style="margin-top:1rem;">
          <strong>Heteroscedasticity confirmed</strong> (p = {bp_pval:.2e} ≪ 0.05)<br>
          OLS Gauss-Markov assumption 4 is violated: residual variance is non-constant.<br>
          <strong>Consequence:</strong> all standard errors, confidence intervals, and
          hypothesis tests from naïve OLS output are <em>invalid</em>.
          Robust standard errors (HC3) are required for inference.<br>
          This does <em>not</em> bias the point estimates (coefficients) —
          but it invalidates every significance test v1 reported.
        </div>
        """, unsafe_allow_html=True)

    show_fig("v2_fig4_ols_residuals.png",
             "Fig 4 · OLS residuals — fan-shaped heteroscedasticity and censoring artefact visible")

    st.markdown("---")

    # ── 4. Overfitting Gap ──────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">4</span>
      <div>
        <div class="section-title-text">Overfitting Gap — What v1 Reported vs What v1 Hid</div>
        <div class="section-desc">
          v1 reported only test R². v2 exposes train R² alongside it.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    rf_train  = diagnostics["rf_train_r2"]
    rf_test   = diagnostics["rf_test_r2"]
    xgb_train = diagnostics["xgb_train_r2"]
    xgb_test  = diagnostics["xgb_test_r2"]

    fig_gap = go.Figure()
    for label, tr, te, ct, ce in [
        ("Random Forest", rf_train,  rf_test,  "#1e4a8a", C_BLUE),
        ("XGBoost",       xgb_train, xgb_test, "#0d5c3e", C_GREEN),
    ]:
        fig_gap.add_trace(go.Bar(
            name=f"{label} Train", x=[label], y=[tr],
            marker=dict(color=ct, line=dict(width=0)),
            text=[f"Train {tr:.4f}"], textposition="inside",
            textfont=dict(color="#fff", size=11),
        ))
        fig_gap.add_trace(go.Bar(
            name=f"{label} Test", x=[label], y=[te],
            marker=dict(color=ce, line=dict(width=0)),
            text=[f"Test {te:.4f}"], textposition="inside",
            textfont=dict(color="#fff", size=11),
        ))

    fig_gap.update_layout(
        barmode="group",
        yaxis=dict(range=[0.75, 1.01]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    )
    chart_style(fig_gap, title="Train R² vs Test R² — Overfitting Exposure", height=360)
    st.plotly_chart(fig_gap, use_container_width=True)

    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(99,179,237,0.2);">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:0.85rem;
                      font-weight:700;color:#63b3ed;margin-bottom:0.8rem;">
            Random Forest
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.88rem;
                      padding:0.35rem 0;border-bottom:1px solid rgba(99,179,237,0.08);">
            <span style="color:#64748b;">Train R²</span>
            <span style="font-family:'Space Grotesk',sans-serif;
                         font-weight:700;color:#f1f5f9;">{rf_train:.4f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.88rem;
                      padding:0.35rem 0;border-bottom:1px solid rgba(99,179,237,0.08);">
            <span style="color:#64748b;">Test R²</span>
            <span style="font-family:'Space Grotesk',sans-serif;
                         font-weight:700;color:#63b3ed;">{rf_test:.4f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.88rem;
                      padding:0.35rem 0;">
            <span style="color:#64748b;">Gap</span>
            <span style="font-family:'Space Grotesk',sans-serif;
                         font-weight:700;color:#f87171;">−{rf_train - rf_test:.4f}</span>
          </div>
          <div style="font-size:0.78rem;color:#475569;margin-top:0.7rem;">
            Root cause: <code>max_depth=None</code> — trees grow until every leaf is pure,
            memorising noise in training data.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with oc2:
        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(52,211,153,0.2);">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:0.85rem;
                      font-weight:700;color:#34d399;margin-bottom:0.8rem;">
            XGBoost
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.88rem;
                      padding:0.35rem 0;border-bottom:1px solid rgba(52,211,153,0.08);">
            <span style="color:#64748b;">Train R²</span>
            <span style="font-family:'Space Grotesk',sans-serif;
                         font-weight:700;color:#f1f5f9;">{xgb_train:.4f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.88rem;
                      padding:0.35rem 0;border-bottom:1px solid rgba(52,211,153,0.08);">
            <span style="color:#64748b;">Test R²</span>
            <span style="font-family:'Space Grotesk',sans-serif;
                         font-weight:700;color:#34d399;">{xgb_test:.4f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.88rem;
                      padding:0.35rem 0;">
            <span style="color:#64748b;">Gap</span>
            <span style="font-family:'Space Grotesk',sans-serif;
                         font-weight:700;color:#fbbf24;">−{xgb_train - xgb_test:.4f}</span>
          </div>
          <div style="font-size:0.78rem;color:#475569;margin-top:0.7rem;">
            Smaller gap due to built-in regularisation (L1/L2 penalties,
            <code>subsample</code>, <code>colsample_bytree</code>).
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="callout callout-red">
      <strong>The gap that v1 hid:</strong> Random Forest train R² = {rf_train:.4f} vs
      test R² = {rf_test:.4f} — a gap of {rf_train - rf_test:.4f}.
      v1 presented {rf_test:.4f} as if it were a clean, generalisable result.
      v2 flags this as a model reliability concern and reports both numbers.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 5. Cross-Validation ─────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">5</span>
      <div>
        <div class="section-title-text">5-Fold Cross-Validation — Stability Check</div>
        <div class="section-desc">
          Confirms that test-set performance is not a lucky split
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-green">
      Cross-validation repeats the train/test split 5 times with different folds.
      A model that scores well on one split but poorly on others is unreliable.
      Low standard deviation across folds = stable, generalisable model.
    </div>
    """, unsafe_allow_html=True)

    show_fig("v2_fig9_cross_validation.png",
             "Fig 9 · 5-fold cross-validation scores — mean ± 1 std for all 9 models")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 · SPATIAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:

    st.markdown("""
    <div class="page-hero">
      <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:1.5px;color:#475569;margin-bottom:0.5rem;">Spatial Analysis</div>
      <div class="gradient-text" style="font-size:2rem;font-weight:700;letter-spacing:-0.4px;">
        Geography, Residuals &amp; The Information Ceiling
      </div>
      <div class="hero-subtitle">
        Moran's I test for spatial autocorrelation · Region classification fix ·
        Quantifying the 15% unexplained variance that no model can recover
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Moran's I ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">I</span>
      <div>
        <div class="section-title-text">Moran's I — Spatial Autocorrelation of Residuals</div>
        <div class="section-desc">
          Tests whether prediction errors cluster geographically
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-purple">
      If residuals are spatially autocorrelated, the model systematically over-predicts
      in some geographic regions and under-predicts in others — a direct signal that
      <strong>location-specific information is missing</strong> from the feature set.
      <br><br>
      The correct methodological response is <strong>Geographically Weighted Regression (GWR)</strong>
      [Fotheringham, Brunsdon &amp; Charlton, 2002], which allows each coefficient to vary by
      location. Expected R² gain: <strong>+0.02–0.04</strong>.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        show_fig("v2_fig11_spatial_residuals.png",
                 "Fig 11 · Spatial residual map — geographic clustering of prediction errors")
    with col2:
        show_fig("v2_fig10_region_income.png",
                 "Fig 10 · Region-income analysis — v1 hardcoded vs v2 percentile thresholds")

    st.markdown("---")

    # ── Region fix ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">FIX</span>
      <div>
        <div class="section-title-text">Region Classification: v1 Bug → v2 Fix</div>
        <div class="section-desc">
          Hardcoded thresholds vs data-driven percentiles
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("""
        <div class="glass-card" style="border-color:rgba(248,113,113,0.2);">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:0.82rem;font-weight:700;
                      color:#f87171;margin-bottom:0.8rem;text-transform:uppercase;letter-spacing:0.8px;">
            ❌  v1 — Broken
          </div>
          <div class="code-block">LOW  = income &lt; 2.0
MID  = 2.0 ≤ income &lt; 5.0
HIGH = income ≥ 5.0  # hardcoded</div>
          <div style="font-size:0.8rem;color:#64748b;margin-top:0.7rem;">
            These thresholds are arbitrary constants with no statistical basis.
            Change the dataset or time period and the classification breaks entirely.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with rc2:
        st.markdown("""
        <div class="glass-card" style="border-color:rgba(52,211,153,0.2);">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:0.82rem;font-weight:700;
                      color:#34d399;margin-bottom:0.8rem;text-transform:uppercase;letter-spacing:0.8px;">
            ✅  v2 — Fixed
          </div>
          <div class="code-block">LOW  = income &lt; 33rd percentile
MID  = 33rd – 67th percentile
HIGH = &gt; 67th percentile</div>
          <div style="font-size:0.8rem;color:#64748b;margin-top:0.7rem;">
            Thresholds computed from the data. Remain valid under any income distribution,
            guarantee equal-ish group sizes, and generalise to any dataset.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Information ceiling ─────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <span class="section-num">15%</span>
      <div>
        <div class="section-title-text">The Information Ceiling</div>
        <div class="section-desc">
          Why R² ≈ 0.85 is the true limit — and what it would take to break through it
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-amber">
      XGBoost achieves R² ≈ 0.85 on this dataset. The remaining 15% unexplained variance
      <strong>cannot be recovered by better methodology</strong> — it is structurally absent
      from the 1990 Census. The table below quantifies each component and the specific
      intervention that would lift R² further.
    </div>
    """, unsafe_allow_html=True)

    ceiling = pd.DataFrame({
        "Source of Unexplained Variance": [
            "Target censored at $500,001 (4.7% of data)",
            "Missing: school quality / district ratings",
            "Missing: crime &amp; safety data",
            "Missing: zoning &amp; land-use data",
            "Within-block-group heterogeneity",
        ],
        "Est. R² Loss": ["0.01–0.02", "0.03–0.05", "0.01–0.03", "0.01–0.02", "0.02–0.04"],
        "Methodological Remedy": [
            "Tobit regression (models censored response directly)",
            "Enrich feature set with external school-rating data",
            "Enrich with crime indices (FBI UCR, local PD data)",
            "Enrich with GIS zoning / land-use layers",
            "Smaller geographic unit of analysis if available",
        ],
        "Expected R² Gain": ["+0.01–0.02", "+0.03–0.06", "+0.01–0.03", "+0.01–0.02", "+0.01–0.02"],
    })
    st.dataframe(ceiling, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="callout callout-blue">
      <strong>Key principle</strong> [Hand, 2006]: a higher R² does not always mean a better model.
      A model scoring 0.87 on an enriched dataset is more useful than one scoring 0.87 by
      overfitting a degraded one. The value of v2 is not a higher number —
      it is a <em>defensible</em> number with honest uncertainty quantification.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 · LIVE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab5:

    st.markdown(f"""
    <div class="page-hero">
      <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:1.5px;color:#475569;margin-bottom:0.5rem;">Live Prediction</div>
      <div class="gradient-text" style="font-size:2rem;font-weight:700;letter-spacing:-0.4px;">
        XGBoost · Best Model
      </div>
      <div class="hero-subtitle">
        R² = {best_row['r2']:.4f} · Log RMSE = {best_row['rmse_log']:.4f} ·
        Dollar RMSE = ${best_row['rmse_dollar']:,.0f} (Jensen-corrected) ·
        Adjust the sliders below and the prediction updates instantly.
      </div>
    </div>
    """, unsafe_allow_html=True)

    form_col, result_col = st.columns([1.1, 0.9], gap="large")

    with form_col:
        st.markdown("""
        <div style="font-family:'Space Grotesk',sans-serif;font-size:0.9rem;font-weight:700;
                    color:#f1f5f9;margin-bottom:1rem;padding-bottom:0.5rem;
                    border-bottom:1px solid rgba(99,179,237,0.12);">
          Block Group Characteristics
        </div>
        """, unsafe_allow_html=True)

        longitude          = st.slider("Longitude", -124.35, -114.31, -122.00, step=0.01,
                                       help="Western CA ≈ −124 · Eastern CA ≈ −114")
        latitude           = st.slider("Latitude",    32.54,   41.95,   37.50, step=0.01,
                                       help="Southern CA ≈ 32.5 · Northern CA ≈ 42")
        housing_median_age = st.slider("Median Housing Age (years)", 1, 52, 28,
                                       help="Age of the median housing unit in the block group")
        median_income      = st.slider("Median Income (×$10,000)", 0.50, 15.00, 3.50, step=0.10,
                                       help="3.5 → median household income of $35,000")

        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:1px;color:#475569;margin:0.7rem 0 0.4rem;">
          Counts
        </div>""", unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        total_rooms    = cc1.number_input("Total Rooms",    2,    39320, 2500, step=50)
        total_bedrooms = cc2.number_input("Total Bedrooms", 1,     6445,  500, step=10)
        population     = cc1.number_input("Population",    10,    35682, 1200, step=50)
        households     = cc2.number_input("Households",     5,     6082,  400, step=10)

    with result_col:
        # ── Feature engineering ──────────────────────────────────────────
        p99       = diagnostics.get("pop_per_hh_p99", 10.0)
        hh        = max(households, 1)
        rooms_per_hh = total_rooms / hh
        bed_per_room = total_bedrooms / max(total_rooms, 1)
        pop_per_hh   = min(population / hh, p99)

        input_dict = {
            "longitude":                longitude,
            "latitude":                 latitude,
            "housing_median_age":       housing_median_age,
            "total_rooms":              total_rooms,
            "total_bedrooms":           total_bedrooms,
            "population":               population,
            "households":               households,
            "median_income":            median_income,
            "rooms_per_household":      rooms_per_hh,
            "bedrooms_per_room":        bed_per_room,
            "population_per_household": pop_per_hh,
        }
        X_ordered = np.array([[input_dict[f] for f in features_tree]])

        log_pred          = model.predict(X_ordered)[0]
        sigma2_xgb        = sigma2["sigma2_xgb"]
        pred_corrected    = np.exp(log_pred + sigma2_xgb / 2)
        pred_uncorrected  = np.exp(log_pred)
        jensen_correction = pred_corrected - pred_uncorrected

        # ── Prediction box ────────────────────────────────────────────────
        st.markdown(f"""
        <div class="prediction-box">
          <div class="pred-label">Predicted Median House Value</div>
          <div class="pred-amount">${pred_corrected:,.0f}</div>
          <div style="font-size:0.75rem;color:#475569;margin-top:0.5rem;">
            Jensen's inequality bias correction applied [Duan, 1983]
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # ── Jensen breakdown ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(167,139,250,0.2);">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:0.82rem;font-weight:700;
                      color:#a78bfa;margin-bottom:0.8rem;text-transform:uppercase;
                      letter-spacing:0.8px;">Jensen Correction Breakdown</div>
          <div style="display:flex;flex-direction:column;gap:0.5rem;font-size:0.86rem;">
            <div style="display:flex;justify-content:space-between;
                        padding:0.35rem 0;border-bottom:1px solid rgba(167,139,250,0.08);">
              <span style="color:#64748b;">Raw log prediction ŷ</span>
              <span style="font-family:'Space Grotesk',sans-serif;
                           font-weight:600;color:#94a3b8;">{log_pred:.4f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;
                        padding:0.35rem 0;border-bottom:1px solid rgba(167,139,250,0.08);">
              <span style="color:#64748b;">Naïve exp(ŷ)</span>
              <span style="font-family:'Space Grotesk',sans-serif;
                           font-weight:600;color:#f87171;">${pred_uncorrected:,.0f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;
                        padding:0.35rem 0;border-bottom:1px solid rgba(167,139,250,0.08);">
              <span style="color:#64748b;">Correction exp(σ²/2) adds</span>
              <span style="font-family:'Space Grotesk',sans-serif;
                           font-weight:600;color:#fbbf24;">+${jensen_correction:,.0f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:0.35rem 0;">
              <span style="color:#e2e8f0;font-weight:600;">Corrected exp(ŷ + σ²/2)</span>
              <span style="font-family:'Space Grotesk',sans-serif;
                           font-weight:700;color:#34d399;">${pred_corrected:,.0f}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ── Engineered features ───────────────────────────────────────────
        st.markdown("""
        <div style="font-family:'Space Grotesk',sans-serif;font-size:0.78rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:1px;color:#475569;margin:0.5rem 0 0.3rem;">
          Engineered Features (computed from inputs)
        </div>""", unsafe_allow_html=True)

        eng = pd.DataFrame({
            "Feature":  ["rooms_per_household", "bedrooms_per_room", "population_per_household"],
            "Value":    [f"{rooms_per_hh:.3f}", f"{bed_per_room:.3f}", f"{pop_per_hh:.3f}"],
            "Formula":  [
                "total_rooms ÷ households",
                "total_bedrooms ÷ total_rooms",
                f"population ÷ households  ·  winsorised at {p99:.2f}",
            ],
        })
        st.dataframe(eng, use_container_width=True, hide_index=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        show_fig("v2_fig6_rf_importances.png",
                 "Fig 6 · Feature importances (Random Forest) — "
                 "median_income and location dominate; engineered features add signal")
    with col2:
        show_fig("v2_fig3_correlation_matrix.png",
                 "Fig 3 · Correlation matrix — feature relationships driving model selection")

    st.markdown("---")

    st.markdown("""
    <div class="callout callout-amber" style="font-size:0.82rem;">
      <strong>Temporal caveat:</strong> this model is trained on 1990 Census data.
      Predictions reflect 1990 price levels and are not interpretable as current market values.
      The dataset target is censored at $500,001 — predictions near or above this threshold
      carry additional uncertainty not captured by the standard error.
    </div>
    """, unsafe_allow_html=True)
