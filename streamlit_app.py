import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import beta as beta_dist

from src.data.generator import CustomerDataGenerator, DatasetConfig
from src.policy.bandit import ThompsonSamplingBandit, BanditConfig
# ── Auto-train if models missing ──────────────────────────────────────────────
if not Path("models/t_learner.pkl").exists():
    import subprocess, sys
    subprocess.run([sys.executable, "train.py"], check=True)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Causal Decision Engine",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Industrial Engine CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Share Tech Mono', monospace;
        background-color: #0a0a0f;
        color: #e0e0e0;
    }

    .stApp { background-color: #0a0a0f; }

    /* HEADER BANNER */
    .engine-header {
        background: linear-gradient(90deg, #0a0a0f 0%, #0d1f0d 40%, #0a0a0f 100%);
        border: 1px solid #00ff41;
        border-radius: 4px;
        padding: 20px 30px;
        margin-bottom: 24px;
        box-shadow: 0 0 30px rgba(0,255,65,0.15), inset 0 0 30px rgba(0,255,65,0.03);
    }
    .engine-header h1 {
        font-family: 'Orbitron', monospace;
        color: #00ff41;
        font-size: 2rem;
        letter-spacing: 6px;
        text-shadow: 0 0 20px #00ff41;
        margin: 0;
    }
    .engine-header p {
        color: #4a9e4a;
        font-size: 0.75rem;
        letter-spacing: 3px;
        margin: 6px 0 0 0;
    }

    /* PANEL CARDS */
    .panel {
        background: #0d1117;
        border: 1px solid #1a3a1a;
        border-left: 3px solid #00ff41;
        border-radius: 2px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 0 10px rgba(0,255,65,0.05);
    }
    .panel-red {
        border-left: 3px solid #ff4141;
        box-shadow: 0 0 10px rgba(255,65,65,0.05);
    }
    .panel-amber {
        border-left: 3px solid #ffb300;
        box-shadow: 0 0 10px rgba(255,179,0,0.05);
    }
    .panel-blue {
        border-left: 3px solid #00b4ff;
        box-shadow: 0 0 10px rgba(0,180,255,0.05);
    }

    /* RECOMMENDATION BOX */
    .rec-box {
        background: #050f05;
        border: 2px solid #00ff41;
        border-radius: 4px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 40px rgba(0,255,65,0.2), inset 0 0 40px rgba(0,255,65,0.03);
    }
    .rec-label {
        font-family: 'Orbitron', monospace;
        color: #4a9e4a;
        font-size: 0.65rem;
        letter-spacing: 4px;
        margin-bottom: 8px;
    }
    .rec-value {
        font-family: 'Orbitron', monospace;
        color: #00ff41;
        font-size: 1.6rem;
        font-weight: 700;
        text-shadow: 0 0 20px #00ff41;
        margin-bottom: 12px;
    }
    .rec-lift {
        font-family: 'Orbitron', monospace;
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 900;
        text-shadow: 0 0 30px #00ff41;
    }
    .rec-sub {
        color: #4a9e4a;
        font-size: 0.7rem;
        letter-spacing: 2px;
        margin-top: 8px;
    }

    /* STATUS INDICATORS */
    .status-row {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
    }
    .status-pill {
        background: #0d1f0d;
        border: 1px solid #00ff41;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.7rem;
        color: #00ff41;
        letter-spacing: 2px;
    }
    .status-pill-off {
        border-color: #333;
        color: #333;
    }

    /* METRIC READOUT */
    .readout {
        background: #050f05;
        border: 1px solid #1a3a1a;
        border-radius: 2px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .readout-label {
        color: #4a9e4a;
        font-size: 0.65rem;
        letter-spacing: 3px;
        margin-bottom: 4px;
    }
    .readout-value {
        font-family: 'Orbitron', monospace;
        color: #00ff41;
        font-size: 1.3rem;
        text-shadow: 0 0 10px rgba(0,255,65,0.5);
    }

    /* TAB STYLING */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d1117;
        border-bottom: 1px solid #1a3a1a;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Share Tech Mono', monospace;
        color: #4a9e4a;
        letter-spacing: 2px;
        font-size: 0.75rem;
        padding: 12px 24px;
        border-right: 1px solid #1a3a1a;
    }
    .stTabs [aria-selected="true"] {
        background: #0d1f0d !important;
        color: #00ff41 !important;
        border-bottom: 2px solid #00ff41 !important;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: #0a0a0f;
        border-right: 1px solid #1a3a1a;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #4a9e4a !important;
        font-size: 0.7rem;
        letter-spacing: 2px;
    }

    /* DIVIDER */
    hr { border-color: #1a3a1a !important; }

    /* HIDE DEFAULT ELEMENTS */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Make header transparent but keep it alive in DOM */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        border-bottom: none !important;
    }

    /* FORCE sidebar toggle arrow always visible */
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] button,
    section[data-testid="stSidebarCollapsedControl"],
    div[data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        opacity: 1 !important;
        display: flex !important;
        pointer-events: all !important;
        z-index: 999999 !important;
        position: fixed !important;
        top: 0.5rem !important;
        left: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LABELS     = {0: "NO ACTION", 1: "EMAIL CAMPAIGN", 2: "DISCOUNT OFFER"}
ARM_COLORS = {0: "#6b7280",   1: "#00b4ff",        2: "#00ff41"}
COSTS_INR  = {0: 0,           1: 400,              2: 2000}   # ₹
FEATURE_COLS    = ["age", "tenure_months", "monthly_spend", "usage_score", "clv"]
CONFOUNDER_COLS = ["am_quality"]
MODEL_FILES     = {
    "t_learner":     "models/t_learner.pkl",
    "double_ml":     "models/double_ml.pkl",
    "causal_forest": "models/causal_forest.pkl",
}

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_FILES.items():
        if Path(path).exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models

@st.cache_data
def load_dataset():
    path = Path("data/synthetic/customers.csv")
    if path.exists():
        return pd.read_csv(path)
    return CustomerDataGenerator(DatasetConfig(n_samples=5_000)).generate()

@st.cache_resource
def get_bandit():
    return ThompsonSamplingBandit(BanditConfig(n_arms=3, arm_costs=[0, 400, 2000]))

def compute_cate(models, X, W):
    XW   = np.hstack([X, W])
    cate = {0: 0.0}
    if "t_learner" in models:
        arms    = models["t_learner"]
        base    = arms[0].predict_proba(XW)[0, 1]
        cate[1] = float(arms[1].predict_proba(XW)[0, 1] - base)
        cate[2] = float(arms[2].predict_proba(XW)[0, 1] - base)
    else:
        cate[1] = 0.08
        cate[2] = 0.18
    return cate

PLOTLY_THEME = dict(
    template="plotly_dark",
    font=dict(family="Share Tech Mono", color="#4a9e4a", size=11),
)

def apply_bg(fig):
    fig.update_layout(paper_bgcolor="#0a0a0f", plot_bgcolor="#0d1117")
    return fig

# ── Load ──────────────────────────────────────────────────────────────────────
models = load_models()
df     = load_dataset()
bandit = get_bandit()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="engine-header">
    <h1>⚙ CAUSAL DECISION ENGINE</h1>
    <p>HETEROGENEOUS TREATMENT EFFECT ESTIMATOR // VERSION 1.0.0 // ONLINE</p>
</div>
""", unsafe_allow_html=True)

model_status = " ".join([
    f'<span class="status-pill">■ {n.upper().replace("_"," ")}</span>'
    if n in models else
    f'<span class="status-pill status-pill-off">□ {n.upper().replace("_"," ")}</span>'
    for n in ["t_learner", "double_ml", "causal_forest"]
])
st.markdown(f'<div class="status-row">{model_status}</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;color:#00ff41;
                font-size:0.8rem;letter-spacing:4px;margin-bottom:20px;
                text-shadow:0 0 10px #00ff41">
        ⚙ INPUT PANEL
    </div>
    """, unsafe_allow_html=True)

    analyze = st.button("⚡ ANALYZE CUSTOMER", use_container_width=True, type="primary")
    st.markdown("**CUSTOMER PROFILE**")
    age           = st.slider("AGE",                  18,   75,   35)
    tenure_months = st.slider("TENURE (MONTHS)",       1,  120,   24)
    monthly_spend = st.slider("MONTHLY SPEND (₹)",  1500, 150000, 12000)
    usage_score   = st.slider("USAGE SCORE",           0,  100,   28)
    clv           = st.number_input("LIFETIME VALUE (₹)", 5000, 5000000, 288000, step=5000)
    am_quality    = st.slider("AM QUALITY SCORE",    -3.0,  3.0,  0.0, 0.1)

    st.divider()
    st.markdown(f"""
    <div style="font-size:0.65rem;color:#4a9e4a;letter-spacing:2px">
        ARM COSTS<br>
        ├ NO ACTION &nbsp;&nbsp;&nbsp;&nbsp;: ₹0<br>
        ├ EMAIL CAMPAIGN: ₹400<br>
        └ DISCOUNT OFFER: ₹2,000
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙ DECISION CENTER",
    "◈ CATE ANALYSIS",
    "◉ BANDIT ENGINE",
    "▣ MODEL COMPARISON",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DECISION CENTER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    X  = np.array([[age, tenure_months, monthly_spend, usage_score, clv]])
    W  = np.array([[am_quality]])
    cate     = compute_cate(models, X, W)
    cate_arr = np.array([cate.get(i, 0.0) for i in range(3)])
    arm      = bandit.select_arm(cate=cate_arr)
    lift     = float(cate_arr[arm])
    cost_inr = COSTS_INR[arm]
    roi      = (lift * clv - cost_inr) / (cost_inr + 1) * 100

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown(f"""
        <div class="rec-box">
            <div class="rec-label">▶ OPTIMAL INTERVENTION</div>
            <div class="rec-value">{LABELS[arm]}</div>
            <div class="rec-lift">+{lift:.1%}</div>
            <div class="rec-sub">ESTIMATED RETENTION LIFT</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="readout">
            <div class="readout-label">CAUSAL LIFT</div>
            <div class="readout-value">+{lift:.4f}</div>
        </div>
        <div class="readout">
            <div class="readout-label">INTERVENTION COST</div>
            <div class="readout-value">₹{cost_inr:,}</div>
        </div>
        <div class="readout">
            <div class="readout-label">EST. ROI</div>
            <div class="readout-value">{roi:+.1f}%</div>
        </div>
        <div class="readout">
            <div class="readout-label">BANDIT CONFIDENCE</div>
            <div class="readout-value">{bandit.expected_rewards()[arm]:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fig = go.Figure()
        for i in range(3):
            fig.add_trace(go.Bar(
                x=[LABELS[i]],
                y=[cate.get(i, 0.0)],
                name=LABELS[i],
                marker=dict(
                    color=ARM_COLORS[i],
                    line=dict(color=ARM_COLORS[i], width=1),
                    opacity=1.0 if i == arm else 0.4,
                ),
                text=[f"+{cate.get(i,0):.2%}"],
                textposition="outside",
                textfont=dict(color=ARM_COLORS[i], family="Share Tech Mono"),
            ))
        fig.update_layout(
            **PLOTLY_THEME,
            title=dict(text="◈ CATE PER TREATMENT ARM", font=dict(color="#00ff41", size=13, family="Share Tech Mono")),
            yaxis=dict(title="CAUSAL EFFECT (CATE)", gridcolor="#1a3a1a", color="#4a9e4a"),
            xaxis=dict(gridcolor="#1a3a1a", color="#4a9e4a"),
            showlegend=False,
            height=340,
            bargap=0.4,
        )
        fig.add_hline(y=0, line_color="#1a3a1a", line_width=1)
        st.plotly_chart(apply_bg(fig), use_container_width=True)

        st.markdown(f"""
        <div class="panel">
            <span style="color:#00ff41;letter-spacing:2px;font-size:0.7rem">▶ RATIONALE</span><br><br>
            <span style="color:#e0e0e0">'{LABELS[arm]}' produces the highest causal retention lift
            for this customer profile.</span><br><br>
            <span style="color:#4a9e4a;font-size:0.7rem">
            CLV: ₹{clv:,.0f} &nbsp;|&nbsp;
            USAGE: {usage_score}/100 &nbsp;|&nbsp;
            TENURE: {tenure_months}M &nbsp;|&nbsp;
            SPEND: ₹{monthly_spend:,.0f}/mo
            </span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="panel"><span style="color:#00ff41;letter-spacing:3px;font-size:0.75rem">◈ CATE DISTRIBUTION ANALYSIS // 2,000 CUSTOMER SAMPLE</span></div>', unsafe_allow_html=True)

    sample = df.sample(min(2000, len(df)), random_state=42).copy()

    if "t_learner" in models:
        arms      = models["t_learner"]
        XW_sample = sample[FEATURE_COLS + CONFOUNDER_COLS].values
        base      = arms[0].predict_proba(XW_sample)[:, 1]
        sample["cate_email"]    = arms[1].predict_proba(XW_sample)[:, 1] - base
        sample["cate_discount"] = arms[2].predict_proba(XW_sample)[:, 1] - base
        sample["clv_inr"]       = sample["clv"] * 80

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                sample, x="cate_email", nbins=60,
                template="plotly_dark",
            )
            fig.update_traces(marker_color="#00b4ff", marker_line_color="#00b4ff", opacity=0.8)
            fig.add_vline(x=sample["cate_email"].mean(), line_dash="dash",
                          line_color="#00ff41", annotation_text=f"μ={sample['cate_email'].mean():.3f}",
                          annotation_font_color="#00ff41")
            fig.update_layout(
                title=dict(text="EMAIL CAMPAIGN — CATE DISTRIBUTION", font=dict(color="#00b4ff", size=12, family="Share Tech Mono")),
                xaxis=dict(title="CAUSAL EFFECT", gridcolor="#1a1a2e", color="#4a9e4a"),
                yaxis=dict(title="COUNT", gridcolor="#1a1a2e", color="#4a9e4a"),
                height=320,
            )
            st.plotly_chart(apply_bg(fig), use_container_width=True)

        with col2:
            fig = px.histogram(
                sample, x="cate_discount", nbins=60,
                template="plotly_dark",
            )
            fig.update_traces(marker_color="#00ff41", marker_line_color="#00ff41", opacity=0.8)
            fig.add_vline(x=sample["cate_discount"].mean(), line_dash="dash",
                          line_color="#00b4ff", annotation_text=f"μ={sample['cate_discount'].mean():.3f}",
                          annotation_font_color="#00b4ff")
            fig.update_layout(
                title=dict(text="DISCOUNT OFFER — CATE DISTRIBUTION", font=dict(color="#00ff41", size=12, family="Share Tech Mono")),
                xaxis=dict(title="CAUSAL EFFECT", gridcolor="#1a1a2e", color="#4a9e4a"),
                yaxis=dict(title="COUNT", gridcolor="#1a1a2e", color="#4a9e4a"),
                height=320,
            )
            st.plotly_chart(apply_bg(fig), use_container_width=True)

        fig = px.scatter(
            sample, x="clv_inr", y="cate_discount",
            color="usage_score",
            color_continuous_scale=[[0,"#0d1f0d"],[0.5,"#00b4ff"],[1,"#00ff41"]],
            template="plotly_dark",
        )
        fig.update_traces(marker=dict(size=4, opacity=0.7))
        fig.update_layout(
            title=dict(text="DISCOUNT CATE vs CUSTOMER LIFETIME VALUE (₹)", font=dict(color="#00ff41", size=12, family="Share Tech Mono")),
            xaxis=dict(title="CLV (₹)", gridcolor="#1a3a1a", color="#4a9e4a"),
            yaxis=dict(title="CAUSAL EFFECT", gridcolor="#1a3a1a", color="#4a9e4a"),
            coloraxis_colorbar=dict(title="USAGE", tickfont=dict(color="#4a9e4a")),
            height=380,
        )
        st.plotly_chart(apply_bg(fig), use_container_width=True)
    else:
        st.warning("⚠ MODELS NOT LOADED — RUN python train.py FIRST")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BANDIT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="panel"><span style="color:#00ff41;letter-spacing:3px;font-size:0.75rem">◉ THOMPSON SAMPLING BANDIT // REAL-TIME POLICY OPTIMIZER</span></div>', unsafe_allow_html=True)

    state = bandit.state()
    col1, col2, col3 = st.columns(3)
    for i, col in enumerate([col1, col2, col3]):
        with col:
            color = list(ARM_COLORS.values())[i]
            st.markdown(f"""
            <div class="readout" style="border-left:3px solid {color};text-align:center">
                <div class="readout-label" style="color:{color}">{LABELS[i]}</div>
                <div class="readout-value" style="color:{color};font-size:1.8rem">
                    {state['expected_rewards'][i]:.2%}
                </div>
                <div style="color:#4a4a4a;font-size:0.65rem;margin-top:8px">
                    α={state['alpha'][i]:.0f} &nbsp;|&nbsp; β={state['beta'][i]:.0f}<br>
                    COST: ₹{COSTS_INR[i]:,}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    FILL_COLORS = ["rgba(107,114,128,0.15)", "rgba(0,180,255,0.15)", "rgba(0,255,65,0.15)"]
    fig = go.Figure()
    x   = np.linspace(0, 1, 300)
    for i in range(3):
        y = beta_dist.pdf(x, bandit.alpha[i], bandit.beta_[i])
        color = list(ARM_COLORS.values())[i]
        fig.add_trace(go.Scatter(
            x=x, y=y, name=LABELS[i],
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=FILL_COLORS[i],
        ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="BETA POSTERIOR DISTRIBUTIONS PER ARM", font=dict(color="#00ff41", size=12, family="Share Tech Mono")),
        xaxis=dict(title="EXPECTED REWARD", gridcolor="#1a3a1a", color="#4a9e4a"),
        yaxis=dict(title="DENSITY", gridcolor="#1a3a1a", color="#4a9e4a"),
        legend=dict(font=dict(color="#4a9e4a", family="Share Tech Mono")),
        height=380,
    )
    st.plotly_chart(apply_bg(fig), use_container_width=True)

    st.divider()
    st.markdown('<span style="color:#00ff41;letter-spacing:3px;font-size:0.7rem">◉ FEEDBACK INTERFACE</span>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([1, 1, 1])
    with fc1:
        sim_arm = st.selectbox("ARM PULLED", [0, 1, 2], format_func=lambda x: LABELS[x])
    with fc2:
        sim_reward = st.radio("OUTCOME", [1, 0], format_func=lambda x: "✅ RETAINED" if x == 1 else "❌ CHURNED")
    with fc3:
        st.write("")
        st.write("")
        if st.button("⚡ SUBMIT FEEDBACK", use_container_width=True):
            bandit.update(arm=sim_arm, reward=sim_reward)
            st.success(f"BANDIT UPDATED // ARM {sim_arm} // REWARD {sim_reward}")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="panel"><span style="color:#00ff41;letter-spacing:3px;font-size:0.75rem">▣ ESTIMATOR BENCHMARK // CATE COMPARISON</span></div>', unsafe_allow_html=True)

    results = {
        "T-LEARNER\n(EMAIL)":    0.1437,
        "T-LEARNER\n(DISCOUNT)": 0.3164,
        "DOUBLE ML":             0.2445,
        "CAUSAL FOREST DML":     0.2433,
    }
    colors = ["#00b4ff", "#00b4ff", "#ffb300", "#00ff41"]
    fig = go.Figure(go.Bar(
        x=list(results.keys()),
        y=list(results.values()),
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color=colors, width=1)),
        text=[f"{v:.4f}" for v in results.values()],
        textposition="outside",
        textfont=dict(color="#e0e0e0", family="Share Tech Mono"),
    ))
    fig.add_hline(y=0.2445, line_dash="dot", line_color="#ffb300",
                  annotation_text="DOUBLE ML BASELINE",
                  annotation_font_color="#ffb300", annotation_font_size=10)
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="MEAN CATE BY ESTIMATOR", font=dict(color="#00ff41", size=13, family="Share Tech Mono")),
        yaxis=dict(title="MEAN CATE", gridcolor="#1a3a1a", color="#4a9e4a"),
        xaxis=dict(gridcolor="#1a3a1a", color="#4a9e4a"),
        showlegend=False,
        height=380,
        bargap=0.45,
    )
    st.plotly_chart(apply_bg(fig), use_container_width=True)

    st.markdown('<span style="color:#00ff41;letter-spacing:3px;font-size:0.7rem">▣ RETENTION RATES // OBSERVED vs CAUSAL</span>', unsafe_allow_html=True)
    obs = pd.DataFrame({
        "TREATMENT":              ["NO ACTION", "EMAIL CAMPAIGN", "DISCOUNT OFFER"],
        "OBSERVED RATE":          [0.3364,      0.5030,           0.6911],
        "TRUE CAUSAL EFFECT":     [0.0000,      0.6046,           1.4138],
        "COST (₹)":               ["₹0",        "₹400",           "₹2,000"],
    })
    st.dataframe(
        obs.style.set_properties(**{
            "background-color": "#0d1117",
            "color":            "#e0e0e0",
            "border-color":     "#1a3a1a",
            "font-family":      "Share Tech Mono",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("""
    <div class="panel panel-amber" style="margin-top:12px">
        <span style="color:#ffb300;font-size:0.65rem;letter-spacing:2px">⚠ WARNING</span><br>
        <span style="color:#e0e0e0;font-size:0.75rem">
        Observed rates are confounded — high-value customers were preferentially treated.
        CATE estimates correct for selection bias via Double ML cross-fitting.
        </span>
    </div>
    """, unsafe_allow_html=True)