"""
Beyond Linearity v2 — California Housing Analysis
MSc Data Science & Analytics · Streamlit deployment
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ── Must be first Streamlit call ─────────────────────────────────────────────
st.set_page_config(
    page_title="Beyond Linearity v2 · California Housing",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric { background: #f8f9fb; border-radius: 8px; padding: 0.5rem; }
    h1 { color: #1a1a2e; }
    h2 { color: #16213e; border-bottom: 2px solid #e8e8e8; padding-bottom: 0.3rem; }
    .callout-blue  { background:#e8f4fd; border-left:4px solid #1f77b4;
                     padding:0.8rem 1rem; border-radius:4px; margin:0.5rem 0; }
    .callout-green { background:#e8f5e9; border-left:4px solid #2e7d32;
                     padding:0.8rem 1rem; border-radius:4px; margin:0.5rem 0; }
    .callout-warn  { background:#fff3e0; border-left:4px solid #e65100;
                     padding:0.8rem 1rem; border-radius:4px; margin:0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Artifact loader (cached so models load once) ─────────────────────────────
@st.cache_resource(show_spinner="Loading model artefacts…")
def load_artifacts():
    m_dir = BASE / "models"
    missing = [f for f in ["xgb_model.pkl", "sigma2.json", "results_df.csv",
                            "features.json", "diagnostics.json", "dm_results.json"]
               if not (m_dir / f).exists()]
    if missing:
        return None, missing
    model       = joblib.load(m_dir / "xgb_model.pkl")
    sigma2      = json.loads((m_dir / "sigma2.json").read_text())
    results_df  = pd.read_csv(m_dir / "results_df.csv")
    features    = json.loads((m_dir / "features.json").read_text())
    diagnostics = json.loads((m_dir / "diagnostics.json").read_text())
    dm_results  = json.loads((m_dir / "dm_results.json").read_text())
    return (model, sigma2, results_df, features, diagnostics, dm_results), []

artifacts, missing_files = load_artifacts()

if missing_files:
    st.error(
        f"Model artefacts not found: `{', '.join(missing_files)}`\n\n"
        "Run **Section 17** (Save Artifacts) in `Beyond_Linearity_v2.ipynb` first, "
        "then refresh this page."
    )
    st.stop()

model, sigma2, results_df, features, diagnostics, dm_results = artifacts
features_tree   = features["features_tree"]
features_linear = features["features_linear"]


# ── Helper: show a PNG figure if it exists ───────────────────────────────────
def show_fig(name, caption="", width=None):
    p = BASE / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=(width is None))


# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/California_state_seal.svg/200px-California_state_seal.svg.png",
    width=80
)
st.sidebar.title("Beyond Linearity v2")
st.sidebar.caption("California Housing · 1990 Census  \nMSc Data Science & Analytics")
st.sidebar.markdown("---")

PAGES = [
    "🏠  Overview",
    "🏆  Model Leaderboard",
    "🔬  Statistical Diagnostics",
    "🗺️  Spatial Analysis",
    "🔮  Live Prediction",
]
page = st.sidebar.radio("", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
best_row  = results_df.sort_values("rmse_log").iloc[0]
st.sidebar.metric("Best Model",  best_row["name"])
st.sidebar.metric("Best R²",     f"{best_row['r2']:.4f}")
st.sidebar.metric("Best $ RMSE", f"${best_row['rmse_dollar']:,.0f}")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: 1990 CA Census (sklearn fetch_california_housing)  \n"
    "Target censored at $500,001 (4.7%)  \n"
    "20,640 block groups · 80/20 train-test split"
)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 · OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("Beyond Linearity: Rigorous Regression on California Housing Prices")
    st.markdown(
        "**MSc Data Science & Analytics · v2.0 — 14 Methodological Improvements over the Naïve Implementation**"
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Test R²",          f"{best_row['r2']:.4f}",   "XGBoost")
    c2.metric("Models Compared",       "9",                        "OLS → XGBoost")
    c3.metric("Methodological Fixes",  "14",                       "v1 → v2")
    c4.metric("Dataset Size",          "20,640",                   "block groups")

    st.markdown("---")
    st.subheader("The Central Thesis")
    st.markdown(
        "> *\"v1 produced numbers. v2 produces numbers you can **trust**, "
        "interpret in **practical units**, and **defend to a statistician**.\"*"
    )
    st.markdown(
        "Most machine learning tutorials report a test R² and call it done. "
        "This analysis goes further: it runs the statistical tests that expose *why* a model "
        "fails, corrects the back-transformation bias that inflates every dollar-scale prediction, "
        "and honestly documents the information ceiling that no methodology can breach."
    )

    st.markdown("---")
    st.subheader("Three Categories of Improvement (v1 → v2)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Category 1 · Diagnostic Additions")
        st.markdown("""
- **VIF** — Variance Inflation Factor (multicollinearity)
- **Breusch-Pagan test** — heteroscedasticity
- **Diebold-Mariano test** — statistical significance of model differences
- **Moran's I** — spatial autocorrelation of residuals
- **Overfitting gap** — train R² vs test R² for every tree model
        """)
        st.caption("Effect on R²: **none** — but expose hidden model failures v1 silently ignored")
    with col2:
        st.markdown("#### Category 2 · Back-Transformation")
        st.markdown("""
- **Jensen's inequality correction** [Duan, 1983]
- Correct: `exp(ŷ + σ²/2)`
- Naïve: `exp(ŷ)`
- Removes systematic underestimation in dollar predictions
- σ² estimated from training residuals for each model
        """)
        st.caption("Effect on R²: **none** (log-scale R² unchanged) — but dollar RMSE becomes honest")
    with col3:
        st.markdown("#### Category 3 · Data & Hyperparameter Fixes")
        st.markdown("""
- **Winsorisation** — `population_per_household` clipped at 99th percentile (max was 1,243, mean 3.07 — 380 std devs above mean)
- **RidgeCV** — cross-validated α, replacing hardcoded α = 10.0
- **LassoCV** — cross-validated α, replacing hardcoded α = 0.01
        """)
        st.caption("Effect on R²: **direct** for the affected models (Ridge, Lasso)")

    st.markdown("---")
    st.subheader("v1 vs v2 — Full Model Comparison")

    cmp = pd.DataFrame({
        "Model":            ["OLS", "Ridge", "Lasso", "Spline", "GAM",
                             "Neural Net", "Random Forest", "Gradient Boosting", "XGBoost"],
        "v1 R²":            [0.6330, 0.6330, 0.6185, 0.7175, 0.7591,
                             0.7452, 0.8228, 0.8368, 0.8505],
        "v2 Expected R²":   ["≈ 0.633", "≈ 0.633", "≈ 0.620", "≈ 0.718", "≈ 0.760",
                             "≈ 0.746", "≈ 0.822", "≈ 0.837", "≈ 0.851"],
        "Primary Change":   [
            "Category 1 only — diagnostics added, no R² shift",
            "Category 3: RidgeCV selected α ≈ hardcoded value → negligible shift",
            "Category 3: LassoCV tightened regularisation slightly",
            "Category 1 only",
            "Category 1 only",
            "Category 1 only",
            "Category 1 only — but overfitting gap (train 0.97 vs test 0.82) now exposed",
            "Category 1 only",
            "Category 1 only — winner unchanged, result now statistically validated",
        ],
    })
    st.dataframe(cmp, use_container_width=True, hide_index=True)

    st.info(
        "**Why are the R² values so similar?** The 15% unexplained variance is an "
        "*information ceiling* imposed by data the 1990 Census never collected: school "
        "quality, crime rates, zoning, proximity to employment. Methodology cannot recover "
        "information that was never in the data. See the **Spatial Analysis** page for a "
        "quantified breakdown and the interventions that would actually lift R²."
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        show_fig("v2_fig1_target_distribution.png",
                 "Fig 1 · Target distribution — log-normal shape; 4.7% of values censored at $500,001")
    with col2:
        show_fig("v2_fig2_ceiling_effect.png",
                 "Fig 2 · Ceiling effect — censoring artefact visible at top of price range")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 · MODEL LEADERBOARD
# ═════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("Model Leaderboard")
    st.caption(
        "9 models ranked by log-scale RMSE · Dollar RMSE uses Jensen's inequality "
        "bias correction [Duan, 1983] · All results on held-out 20% test set"
    )
    st.markdown("---")

    rdf = results_df.sort_values("rmse_log").reset_index(drop=True)
    best = rdf.iloc[0]
    st.markdown(
        f'<div class="callout-green"><strong>Winner: {best["name"]}</strong> — '
        f'R² = {best["r2"]:.4f} | Log RMSE = {best["rmse_log"]:.4f} | '
        f'Dollar RMSE = ${best["rmse_dollar"]:,.0f} | Dollar MAE = ${best["mae_dollar"]:,.0f}</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Interactive Plotly leaderboard
    st.subheader("R² and RMSE — All Models")
    col1, col2 = st.columns(2)

    with col1:
        colors = ["#2e7d32" if n == best["name"] else "#1f77b4" for n in rdf["name"]]
        fig_r2 = go.Figure(go.Bar(
            x=rdf["r2"], y=rdf["name"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in rdf["r2"]],
            textposition="outside",
            hovertemplate="%{y}: R² = %{x:.4f}<extra></extra>",
        ))
        fig_r2.update_layout(
            title="Test R² (higher = better)",
            xaxis=dict(range=[0.55, 0.92], title="R²"),
            yaxis=dict(title=""),
            height=380,
            margin=dict(l=10, r=60, t=40, b=10),
            plot_bgcolor="#fafafa",
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        fig_rmse = go.Figure(go.Bar(
            x=rdf["rmse_dollar"] / 1000, y=rdf["name"],
            orientation="h",
            marker_color=colors,
            text=[f"${v:,.0f}k" for v in rdf["rmse_dollar"] / 1000],
            textposition="outside",
            hovertemplate="%{y}: $%{x:.1f}k RMSE<extra></extra>",
        ))
        fig_rmse.update_layout(
            title="Dollar RMSE — Jensen-corrected (lower = better)",
            xaxis=dict(title="RMSE ($000s)"),
            yaxis=dict(title=""),
            height=380,
            margin=dict(l=10, r=80, t=40, b=10),
            plot_bgcolor="#fafafa",
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Results Table")
    disp = rdf.copy()
    disp["rmse_dollar"] = disp["rmse_dollar"].apply(lambda x: f"${x:,.0f}")
    disp["mae_dollar"]  = disp["mae_dollar"].apply(lambda x: f"${x:,.0f}")
    disp["r2"]          = disp["r2"].round(4)
    disp["rmse_log"]    = disp["rmse_log"].round(4)
    disp = disp.rename(columns={
        "name": "Model", "rmse_log": "Log RMSE", "r2": "R²",
        "rmse_dollar": "RMSE ($)", "mae_dollar": "MAE ($)"
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Diebold-Mariano Test — Statistical Significance vs OLS Baseline")
    st.caption(
        "Tests whether each model's forecast errors are statistically distinguishable from OLS. "
        "p < 0.05 = significant improvement. A higher R² that is *not* DM-significant means "
        "the improvement may be sampling noise. [Harvey, Leybourne & Newbold, 1997]"
    )

    dm_rows = []
    for mname, res in dm_results.items():
        sig  = res["p_value"] < 0.05
        dm_rows.append({
            "Model":        mname,
            "DM Statistic": round(res["dm_stat"], 3),
            "p-value":      round(res["p_value"], 4),
            "Verdict":      "✅ Significantly better than OLS" if sig else "⚠️ Not statistically distinguishable from OLS",
        })
    dm_df = pd.DataFrame(dm_rows).sort_values("p-value")
    st.dataframe(dm_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        show_fig("v2_fig8_predicted_actual.png",
                 "Fig 8 · Predicted vs Actual — OLS (left) vs best model (right)")
    with col2:
        show_fig("v2_fig9_cross_validation.png",
                 "Fig 9 · 5-fold cross-validation — all models (mean ± std)")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 · STATISTICAL DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("Statistical Diagnostics")
    st.caption(
        "The tests v1 skipped — each one exposes a different failure mode "
        "that a naïve implementation silently ignores"
    )
    st.markdown("---")

    # ── 1. VIF ───────────────────────────────────────────────────────────────
    st.subheader("1 · Variance Inflation Factor — Multicollinearity Audit")
    st.markdown(
        "VIF measures how much a predictor's variance is inflated by collinearity with others. "
        "VIF > 10 = severe; VIF > 5 = investigate. Explains **why Ridge R² ≈ OLS R²**: "
        "if multicollinearity were the bottleneck, Ridge would have helped. It didn't — "
        "so the bottleneck is nonlinearity, not collinear predictors."
    )

    vif_df = pd.DataFrame(diagnostics["vif"])
    vif_df["VIF"] = vif_df["VIF"].round(2)

    def vif_flag(v):
        if v > 10:  return "🔴 Severe"
        if v > 5:   return "🟡 Moderate"
        return "🟢 OK"

    vif_df["Status"] = vif_df["VIF"].apply(vif_flag)
    vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(vif_df, use_container_width=True, hide_index=True)
    with col2:
        fig_vif = go.Figure(go.Bar(
            x=vif_df["VIF"],
            y=vif_df["Feature"],
            orientation="h",
            marker_color=["#d32f2f" if v > 10 else "#f57c00" if v > 5 else "#388e3c"
                          for v in vif_df["VIF"]],
            text=[f"{v:.1f}" for v in vif_df["VIF"]],
            textposition="outside",
        ))
        fig_vif.add_vline(x=5,  line_dash="dash", line_color="#f57c00",
                          annotation_text="VIF=5")
        fig_vif.add_vline(x=10, line_dash="dash", line_color="#d32f2f",
                          annotation_text="VIF=10")
        fig_vif.update_layout(
            title="VIF by Feature",
            xaxis_title="VIF", yaxis_title="",
            height=380, margin=dict(l=10, r=60, t=40, b=10),
            plot_bgcolor="#fafafa"
        )
        st.plotly_chart(fig_vif, use_container_width=True)

    st.markdown("---")

    # ── 2. Breusch-Pagan ─────────────────────────────────────────────────────
    st.subheader("2 · Breusch-Pagan Test — Heteroscedasticity")
    bp_stat = diagnostics["bp_stat"]
    bp_pval = diagnostics["bp_pval"]

    col1, col2, col3 = st.columns(3)
    col1.metric("BP Statistic", f"{bp_stat:.4f}")
    col2.metric("p-value",      f"{bp_pval:.6f}")
    col3.metric("Verdict",
                "Heteroscedastic ⚠️" if bp_pval < 0.05 else "Homoscedastic ✅")

    if bp_pval < 0.05:
        st.warning(
            f"**Heteroscedasticity confirmed** (p = {bp_pval:.6f} ≪ 0.05).  \n"
            "OLS Gauss-Markov assumption 4 is violated: residual variance is non-constant.  \n"
            "Consequence: standard errors, confidence intervals, and hypothesis tests derived "
            "from OLS are **invalid**. Robust standard errors (HC3) are required for inference.  \n"
            "This does *not* bias the point estimates (coefficients), but it invalidates "
            "all significance tests reported by naive OLS output."
        )
    show_fig("v2_fig4_ols_residuals.png",
             "Fig 4 · OLS residuals — fan-shaped heteroscedasticity and censoring artefact visible")

    st.markdown("---")

    # ── 3. Overfitting gap ───────────────────────────────────────────────────
    st.subheader("3 · Overfitting Gap — What v1 Reported vs What v1 Hid")
    st.markdown(
        "v1 reported only **test R²**. v2 exposes train R² alongside it. "
        "A large gap means the model memorised training data — the reported test R² "
        "cannot be trusted to generalise."
    )

    rf_train  = diagnostics["rf_train_r2"]
    rf_test   = diagnostics["rf_test_r2"]
    xgb_train = diagnostics["xgb_train_r2"]
    xgb_test  = diagnostics["xgb_test_r2"]

    # Interactive grouped bar
    fig_gap = go.Figure()
    for label, tr, te, color_tr, color_te in [
        ("Random Forest",    rf_train,  rf_test,  "#1565c0", "#90caf9"),
        ("XGBoost",          xgb_train, xgb_test, "#2e7d32", "#a5d6a7"),
    ]:
        fig_gap.add_trace(go.Bar(
            name=f"{label} Train",
            x=[label], y=[tr],
            marker_color=color_tr,
            text=[f"Train: {tr:.4f}"], textposition="inside",
        ))
        fig_gap.add_trace(go.Bar(
            name=f"{label} Test",
            x=[label], y=[te],
            marker_color=color_te,
            text=[f"Test: {te:.4f}"], textposition="inside",
        ))

    fig_gap.update_layout(
        barmode="group",
        title="Train R² vs Test R² — Overfitting Exposure",
        yaxis=dict(title="R²", range=[0.75, 1.01]),
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#fafafa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("RF Train R²",  f"{rf_train:.4f}")
    col1.metric("RF Test R²",   f"{rf_test:.4f}",
                delta=f"{rf_test - rf_train:.4f} gap", delta_color="inverse")
    col2.metric("XGB Train R²", f"{xgb_train:.4f}")
    col2.metric("XGB Test R²",  f"{xgb_test:.4f}",
                delta=f"{xgb_test - xgb_train:.4f} gap", delta_color="inverse")

    st.warning(
        f"**Random Forest gap = {rf_train - rf_test:.4f}** (train {rf_train:.4f} → test {rf_test:.4f}).  \n"
        "Root cause: `max_depth=None` allows unlimited tree growth, perfectly fitting training data.  \n"
        "v1 presented {:.4f} as if it were a clean, generalisable result. "
        "v2 flags this as a model reliability concern and reports both numbers.".format(rf_test)
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 · SPATIAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("Spatial Analysis")
    st.caption(
        "Moran's I test for residual clustering · Geographic error maps · "
        "The information ceiling quantified"
    )
    st.markdown("---")

    st.subheader("Moran's I — Spatial Autocorrelation of Residuals")
    st.markdown(
        "If residuals are spatially correlated, the model is systematically "
        "over-predicting in some geographic regions and under-predicting in others — "
        "a signal that *location-specific* information is missing from the feature set.  \n\n"
        "The correct methodological response is **Geographically Weighted Regression (GWR)** "
        "[Fotheringham, Brunsdon & Charlton, 2002], which allows each coefficient to vary "
        "by location. Expected R² gain: +0.02–0.04."
    )

    col1, col2 = st.columns(2)
    with col1:
        show_fig("v2_fig11_spatial_residuals.png",
                 "Fig 11 · Spatial residual map — geographic clustering of prediction errors")
    with col2:
        show_fig("v2_fig10_region_income.png",
                 "Fig 10 · Region-income analysis — v1 hardcoded thresholds vs v2 percentile-based")

    st.markdown("---")
    st.subheader("The Information Ceiling — Why R² ≈ 0.85 Is the True Limit")
    st.markdown(
        "XGBoost achieves R² ≈ 0.85 on this dataset. The remaining 15% unexplained variance "
        "cannot be recovered by better methodology: it is structurally absent from the 1990 Census."
    )

    ceiling = pd.DataFrame({
        "Source of Unexplained Variance": [
            "Target censoring at $500,001 (4.7% of data)",
            "Missing: school quality / district ratings",
            "Missing: crime & safety data",
            "Missing: zoning & land-use data",
            "Within-block-group heterogeneity",
        ],
        "Est. R² Loss": ["0.01–0.02", "0.03–0.05", "0.01–0.03", "0.01–0.02", "0.02–0.04"],
        "Methodological Remedy": [
            "Tobit regression (models censored response directly)",
            "Enrich feature set with external school-rating data",
            "Enrich feature set with crime/safety indices",
            "Enrich feature set with GIS zoning layers",
            "Smaller geographic unit of analysis (if available)",
        ],
        "Expected R² Gain": ["+0.01–0.02", "+0.03–0.06", "+0.01–0.03", "+0.01–0.02", "+0.01–0.02"],
    })
    st.dataframe(ceiling, use_container_width=True, hide_index=True)

    st.info(
        "**Key principle** [Hand, 2006]: a higher R² does not always mean a better model. "
        "A model that scores 0.87 on an enriched dataset is more useful than one scoring 0.87 "
        "by overfitting to a degraded one. The value of v2 is not a higher number — "
        "it is a *defensible* number with honest uncertainty quantification."
    )

    st.markdown("---")
    st.subheader("Region Classification: v1 Bug → v2 Fix")
    st.markdown(
        "v1 used hardcoded income thresholds to classify block groups as Low / Mid / High income. "
        "v2 uses percentile-based thresholds computed from the data, "
        "which remain valid under any income distribution. "
        "Fig 10 above shows the classification difference."
    )

    col1, col2 = st.columns(2)
    col1.markdown("**v1 (broken):**")
    col1.code("LOW  = income < 2.0\nMID  = 2.0 ≤ income < 5.0\nHIGH = income ≥ 5.0  # hardcoded")
    col2.markdown("**v2 (fixed):**")
    col2.code("LOW  = income < 33rd percentile\nMID  = 33rd–67th percentile\nHIGH = > 67th percentile")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 · LIVE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.title("Live Prediction — XGBoost (Best Model)")
    st.caption(
        f"R² = {best_row['r2']:.4f} · Log RMSE = {best_row['rmse_log']:.4f} · "
        f"Dollar RMSE = ${best_row['rmse_dollar']:,.0f} (Jensen-corrected)  \n"
        "Based on 1990 California Census data · 80/20 train-test split"
    )
    st.markdown("---")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("Block Group Characteristics")
        st.caption("Adjust sliders to describe a census block group; prediction updates instantly.")

        longitude           = st.slider("Longitude",                -124.35, -114.31, -122.00, step=0.01,
                                        help="Western CA ≈ −124, Eastern ≈ −114")
        latitude            = st.slider("Latitude",                   32.54,   41.95,   37.50, step=0.01,
                                        help="Southern CA ≈ 32.5, Northern ≈ 42")
        housing_median_age  = st.slider("Housing Median Age (years)",      1,      52,      28,
                                        help="Age of the median housing unit in the block group")
        median_income       = st.slider("Median Income (×$10,000)",     0.50,   15.00,    3.50, step=0.10,
                                        help="e.g. 3.5 → median household income of $35,000")

        st.markdown("**Counts**")
        cc1, cc2 = st.columns(2)
        total_rooms    = cc1.number_input("Total Rooms",    2,    39320, 2500, step=50)
        total_bedrooms = cc2.number_input("Total Bedrooms", 1,     6445,  500, step=10)
        population     = cc1.number_input("Population",    10,    35682, 1200, step=50)
        households     = cc2.number_input("Households",     5,     6082,  400, step=10)

    with col_result:
        st.subheader("Prediction")

        # ── Feature engineering (mirrors v2 notebook exactly) ───────────────
        p99         = diagnostics.get("pop_per_hh_p99", 10.0)
        hh          = max(households, 1)
        rooms_per_hh   = total_rooms    / hh
        bed_per_room   = total_bedrooms / max(total_rooms, 1)
        pop_per_hh     = min(population / hh, p99)

        # Build feature dict then align to exact training order
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
        feat_order = features_tree
        X_ordered  = np.array([[input_dict[f] for f in feat_order]])

        log_pred            = model.predict(X_ordered)[0]
        sigma2_xgb          = sigma2["sigma2_xgb"]
        pred_corrected      = np.exp(log_pred + sigma2_xgb / 2)   # Jensen correction
        pred_uncorrected    = np.exp(log_pred)
        jensen_correction   = pred_corrected - pred_uncorrected

        st.metric(
            "Predicted Median House Value",
            f"${pred_corrected:,.0f}",
            help="Jensen's inequality bias correction applied [Duan, 1983]"
        )

        st.markdown(
            f'<div class="callout-blue">'
            f'<strong>Jensen correction breakdown</strong><br>'
            f'Naïve exp(ŷ) = ${pred_uncorrected:,.0f}<br>'
            f'Correction term exp(σ²/2) adds ${jensen_correction:,.0f}<br>'
            f'Corrected value = <strong>${pred_corrected:,.0f}</strong>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("**Engineered Features (computed from inputs)**")
        eng = pd.DataFrame({
            "Feature":    ["rooms_per_household", "bedrooms_per_room", "population_per_household"],
            "Value":      [f"{rooms_per_hh:.3f}", f"{bed_per_room:.3f}", f"{pop_per_hh:.3f}"],
            "Formula":    [
                "total_rooms ÷ households",
                "total_bedrooms ÷ total_rooms",
                f"population ÷ households · winsorised at {p99:.2f} (99th pct)",
            ],
        })
        st.dataframe(eng, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**All 11 Model Inputs**")
        input_display = pd.DataFrame({
            "Feature": feat_order,
            "Value":   [f"{input_dict[f]:.4f}" for f in feat_order],
        })
        st.dataframe(input_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        show_fig("v2_fig6_rf_importances.png",
                 "Fig 6 · Feature importances (Random Forest) — "
                 "median income and location dominate; engineered features add signal")
    with col2:
        show_fig("v2_fig3_correlation_matrix.png",
                 "Fig 3 · Correlation matrix — feature relationships used in model selection")

    st.markdown("---")
    st.caption(
        "⚠️ **Temporal caveat**: this model is trained on 1990 Census data. "
        "Predictions reflect 1990 price levels and are not interpretable as current market values. "
        "The dataset target is also censored at $500,001 — predictions near or above this "
        "threshold carry additional uncertainty."
    )
