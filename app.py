"""
╔══════════════════════════════════════════════════════════════════════╗
║      LUNG CANCER DETECTION  –  ML Intelligence Dashboard            ║
║      Steps 1-10 | Classification | Regression | ARM | Bias          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════
# STEP 1 ─ Import packages
# ─────────────────────────────────────────────────────────────────────
# All required libraries are imported here. Run this cell first.
# ═══════════════════════════════════════════════════════════════════════
import warnings, io, textwrap
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from scipy.stats import chi2_contingency, pointbiserialr
from mlxtend.frequent_patterns import apriori, association_rules

# ─── Page configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Lung Cancer ML Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# GLOBAL CSS  –  crisp light theme with medical-grade aesthetic
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & body ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #FAFBFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1.5px solid #E2E8F0 !important;
}
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stToolbar"]          { display: none !important; }
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* ── Streamlit native overrides ── */
h1,h2,h3,h4,h5,h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #0F172A !important;
}
p, li, td, th { font-family: 'Plus Jakarta Sans', sans-serif !important; }
code, pre      { font-family: 'JetBrains Mono', monospace !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 50%, #0EA5E9 100%);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 80% 50%, rgba(255,255,255,.08) 0%, transparent 60%);
}
.hero-title {
    font-size: 30px; font-weight: 800;
    color: #FFFFFF; letter-spacing: -.5px;
    margin: 0 0 6px;
}
.hero-sub {
    font-size: 14px; color: rgba(255,255,255,.75);
    font-weight: 400; margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.25);
    color: #FFFFFF;
    font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 999px;
    margin-right: 6px; margin-top: 10px;
    letter-spacing: .04em;
}

/* ── KPI cards ── */
.kpi-grid { display: flex; gap: 14px; margin: 0 0 24px; flex-wrap: wrap; }
.kpi {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 22px;
    flex: 1; min-width: 140px;
    box-shadow: 0 1px 6px rgba(0,0,0,.04);
    transition: box-shadow .2s, transform .2s;
    position: relative; overflow: hidden;
}
.kpi:hover { box-shadow: 0 8px 24px rgba(37,99,235,.10); transform: translateY(-2px); }
.kpi::after {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 4px; height: 100%;
    border-radius: 4px 0 0 4px;
}
.kpi.blue::after  { background: #2563EB; }
.kpi.green::after { background: #10B981; }
.kpi.amber::after { background: #F59E0B; }
.kpi.red::after   { background: #EF4444; }
.kpi.violet::after{ background: #8B5CF6; }
.kpi.cyan::after  { background: #06B6D4; }
.kpi-label {
    font-size: 11px; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase;
    color: #64748B; margin-bottom: 8px;
}
.kpi-val {
    font-size: 28px; font-weight: 800;
    color: #0F172A; line-height: 1;
}
.kpi-sub { font-size: 11px; color: #94A3B8; margin-top: 4px; }

/* ── Section titles ── */
.sec-title {
    font-size: 17px; font-weight: 800;
    color: #0F172A; margin: 28px 0 14px;
    display: flex; align-items: center; gap: 8px;
}
.sec-title::after {
    content: ''; flex: 1; height: 1.5px;
    background: linear-gradient(90deg, #E2E8F0 0%, transparent 100%);
    border-radius: 2px;
}

/* ── Info / warn / success boxes ── */
.info-box, .warn-box, .ok-box, .danger-box {
    border-radius: 10px; padding: 13px 17px;
    font-size: 13.5px; margin-bottom: 16px;
    font-weight: 500;
}
.info-box   { background:#EFF6FF; border-left:4px solid #2563EB; color:#1E40AF; }
.warn-box   { background:#FFFBEB; border-left:4px solid #F59E0B; color:#92400E; }
.ok-box     { background:#F0FDF4; border-left:4px solid #22C55E; color:#166534; }
.danger-box { background:#FEF2F2; border-left:4px solid #EF4444; color:#991B1B; }

/* ── Badge pills ── */
.pill {
    display: inline-block; padding: 3px 11px;
    border-radius: 999px; font-size: 12.5px; font-weight: 700;
}
.pill-green  { background:#D1FAE5; color:#065F46; }
.pill-amber  { background:#FEF3C7; color:#92400E; }
.pill-red    { background:#FEE2E2; color:#991B1B; }
.pill-blue   { background:#DBEAFE; color:#1D4ED8; }
.pill-violet { background:#EDE9FE; color:#5B21B6; }

/* ── Step label ── */
.step-label {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%;
    background: #2563EB; color: #FFF;
    font-size: 12px; font-weight: 800;
    margin-right: 8px; flex-shrink: 0;
}

/* ── Styled HTML table ── */
.stbl {
    width: 100%; border-collapse: collapse;
    font-size: 13.5px; background: #FFFFFF;
    border-radius: 12px; overflow: hidden;
    box-shadow: 0 1px 6px rgba(0,0,0,.05);
    margin-bottom: 20px;
}
.stbl thead tr { background: #F8FAFC; }
.stbl th {
    padding: 11px 15px; text-align: left;
    font-size: 11px; font-weight: 800;
    letter-spacing: .07em; text-transform: uppercase;
    color: #475569; border-bottom: 1.5px solid #E2E8F0;
}
.stbl td {
    padding: 10px 15px; color: #334155;
    border-bottom: 1px solid #F1F5F9;
}
.stbl tr:last-child td { border-bottom: none; }
.stbl tr:hover td { background: #F8FAFC; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { gap: 6px; background: #F1F5F9; padding: 4px; border-radius: 12px; }
[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13.5px !important; font-weight: 600 !important;
    color: #64748B !important;
    padding: 8px 18px !important;
    transition: all .2s !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #FFFFFF !important;
    color: #2563EB !important;
    box-shadow: 0 1px 6px rgba(0,0,0,.08) !important;
}
[data-testid="stTabPanel"] { padding-top: 20px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] .sidebar-section {
    background: #F8FAFC; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 14px;
}
.sidebar-header {
    font-size: 10px; font-weight: 800;
    letter-spacing: .12em; text-transform: uppercase;
    color: #94A3B8; margin-bottom: 10px;
}

/* ── Correlation badge gradient ── */
.corr-bar {
    height: 6px; border-radius: 3px; margin-top: 4px;
    background: linear-gradient(90deg, #DBEAFE, #2563EB);
}

/* ── Scrollable table wrapper ── */
.scroll-wrap { max-height: 420px; overflow-y: auto; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# COLOUR PALETTE  (consistent across all charts)
# ═══════════════════════════════════════════════════════════════════════
C = {
    "primary"  : "#2563EB",
    "success"  : "#10B981",
    "warning"  : "#F59E0B",
    "danger"   : "#EF4444",
    "violet"   : "#8B5CF6",
    "cyan"     : "#06B6D4",
    "slate"    : "#64748B",
    "bg"       : "#FFFFFF",
    "grid"     : "#F1F5F9",
    "text"     : "#0F172A",
    "yes_clr"  : "#EF4444",
    "no_clr"   : "#10B981",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
    font=dict(family="Plus Jakarta Sans", color=C["text"], size=12),
    margin=dict(l=10, r=10, t=48, b=10),
    hoverlabel=dict(bgcolor="#FFFFFF", font_size=12,
                    font_family="Plus Jakarta Sans",
                    bordercolor="#E2E8F0"),
)

MODEL_COLOURS = {
    "Decision Tree"       : C["primary"],
    "Random Forest"       : C["success"],
    "Gradient Boosting"   : C["warning"],
    "Logistic Regression" : C["violet"],
    "SVM"                 : C["cyan"],
    "KNN"                 : "#F97316",
    "Naive Bayes"         : "#EC4899",
}


# ═══════════════════════════════════════════════════════════════════════
# ██  DATA PIPELINE  (Steps 2 – 6, cached)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def build_pipeline(csv_bytes: bytes):
    """
    STEP 2 – Basic data check
    STEP 3 – Null value handling
    STEP 4 – Label encoding
    STEP 5 – Feature / label split
    STEP 6 – Train / test split (80:20, stratified)
    Returns everything downstream tabs need.
    """

    # ── Step 2: Load & inspect ────────────────────────────────────────
    df_raw = pd.read_csv(io.BytesIO(csv_bytes))
    df_raw.columns = df_raw.columns.str.strip()

    basic = {
        "shape"      : df_raw.shape,
        "dtypes"     : df_raw.dtypes.astype(str).to_dict(),
        "null_before": df_raw.isnull().sum().to_dict(),
        "describe"   : df_raw.describe(include="all"),
    }

    # ── Step 3: Fill nulls ─────────────────────────────────────────────
    df = df_raw.copy()
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    basic["null_after"] = df.isnull().sum().to_dict()

    # ── Step 4: Label encode ──────────────────────────────────────────
    le_map   = {}    # col → LabelEncoder
    mapping  = []    # list of {Column, Original, Encoded}

    df_enc = df.copy()
    for col in df_enc.columns:
        if df_enc[col].dtype == object or str(df_enc[col].dtype) == "str":
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            le_map[col] = le
            for orig, enc in zip(le.classes_, le.transform(le.classes_)):
                mapping.append({"Column": col, "Original Value": orig, "Encoded Value": int(enc)})

    map_df = pd.DataFrame(mapping)

    # ── Step 5: Split features / label ───────────────────────────────
    LABEL = "LUNG_CANCER"
    X = df_enc.drop(columns=[LABEL])
    y = df_enc[LABEL]

    # ── Step 6: Train / Test 80:20 stratified ─────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Correlation & chi-square stats ───────────────────────────────
    corr_series = df_enc.corr()[LABEL].drop(LABEL).abs().sort_values(ascending=False)

    chi2_rows = []
    for col in X.columns:
        ct   = pd.crosstab(df_enc[col], df_enc[LABEL])
        chi2, p, dof, _ = chi2_contingency(ct)
        chi2_rows.append({"Feature": col, "Chi2": round(chi2, 3), "p-value": round(p, 4),
                           "Significant": "✅ Yes" if p < 0.05 else "❌ No"})
    chi2_df = pd.DataFrame(chi2_rows).sort_values("Chi2", ascending=False)

    return dict(
        df_raw=df_raw, df=df, df_enc=df_enc,
        basic=basic, map_df=map_df, le_map=le_map,
        X=X, y=y, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
        LABEL=LABEL, corr=corr_series, chi2_df=chi2_df,
        features=list(X.columns),
    )


# ═══════════════════════════════════════════════════════════════════════
# ██  CLASSIFICATION ENGINE  (Steps 7 – 10, cached)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_classifiers(X_tr, X_te, y_tr, y_te, dt_d, rf_n, gb_n, features):
    """Steps 7-10: Train all models, collect metrics."""

    def _cv(m, Xtr, ytr):
        return cross_val_score(m, Xtr, ytr, cv=StratifiedKFold(5), scoring="accuracy").mean()

    model_defs = {
        "Decision Tree"       : DecisionTreeClassifier(random_state=42, max_depth=dt_d),
        "Random Forest"       : RandomForestClassifier(n_estimators=rf_n, random_state=42),
        "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=gb_n, random_state=42),
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "SVM"                 : SVC(probability=True, random_state=42),
        "KNN"                 : KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes"         : GaussianNB(),
    }

    results = {}
    for name, m in model_defs.items():
        m.fit(X_tr, y_tr)
        yp_tr = m.predict(X_tr)
        yp_te = m.predict(X_te)

        try:   prob_te = m.predict_proba(X_te)[:, 1]
        except: prob_te = None

        results[name] = {
            "model"      : m,
            "train_acc"  : accuracy_score(y_tr, yp_tr),
            "test_acc"   : accuracy_score(y_te, yp_te),
            "precision"  : precision_score(y_te, yp_te, average="weighted", zero_division=0),
            "recall"     : recall_score(   y_te, yp_te, average="weighted", zero_division=0),
            "f1"         : f1_score(       y_te, yp_te, average="weighted", zero_division=0),
            "auc"        : roc_auc_score(y_te, prob_te) if prob_te is not None else None,
            "cm"         : confusion_matrix(y_te, yp_te),
            "yp_te"      : yp_te,
            "prob_te"    : prob_te,
            "cv_mean"    : _cv(m, X_tr, y_tr),
            "imp"        : m.feature_importances_ if hasattr(m, "feature_importances_") else None,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
# ██  ASSOCIATION RULE MINING  (cached)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_arm(df_enc, min_sup, min_conf, min_lift):
    sym_cols = [c for c in df_enc.columns if c != "LUNG_CANCER"]
    df_bin   = df_enc[sym_cols].copy()

    # Symptom cols: 1 = absent, 2 = present → binarise
    for col in sym_cols:
        df_bin[col] = (df_bin[col] >= 2).astype(bool)
    df_bin["LUNG_CANCER_YES"] = (df_enc["LUNG_CANCER"] == 1)

    freq = apriori(df_bin, min_support=min_sup, use_colnames=True, max_len=4)
    if freq.empty:
        return freq, pd.DataFrame()

    rules = association_rules(freq, metric="confidence",
                               min_threshold=min_conf,
                               num_itemsets=len(freq))
    rules = rules[rules["lift"] >= min_lift].copy()
    rules.sort_values("lift", ascending=False, inplace=True)
    rules.reset_index(drop=True, inplace=True)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return freq, rules


# ═══════════════════════════════════════════════════════════════════════
# ██  BIAS ENGINE  (cached)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_bias(df_raw, df_enc, _best_model, label, features):
    X_all    = df_enc[features]
    y_all    = df_enc[label]
    yp_all   = _best_model.predict(X_all)
    records  = []

    def _add(group_col, group_val, mask):
        if mask.sum() < 5:
            return
        acc  = accuracy_score(y_all[mask], yp_all[mask])
        pos  = yp_all[mask].mean()
        tp   = int(((y_all[mask] == 1) & (yp_all[mask] == 1)).sum())
        fn   = int(((y_all[mask] == 1) & (yp_all[mask] == 0)).sum())
        tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0
        records.append({
            "Group Column" : group_col,
            "Group Value"  : str(group_val),
            "N"            : int(mask.sum()),
            "Accuracy"     : acc,
            "Positive Rate": pos,
            "TPR (Recall)" : tpr,
        })

    # Gender
    if "GENDER" in df_raw.columns:
        for g in df_raw["GENDER"].dropna().unique():
            _add("GENDER", g, (df_raw["GENDER"] == g).values)

    # Age groups
    if "AGE" in df_raw.columns:
        bins   = [0, 40, 55, 70, 200]
        labels = ["< 40", "40 – 55", "56 – 70", "> 70"]
        df_tmp = df_raw.copy()
        df_tmp["_ag"] = pd.cut(df_tmp["AGE"], bins=bins, labels=labels)
        for ag in labels:
            mask = (df_tmp["_ag"] == ag).values
            _add("AGE GROUP", ag, mask)

    # Smoking
    if "SMOKING" in df_raw.columns:
        for sv in df_raw["SMOKING"].dropna().unique():
            lbl = "Smoker" if sv == 2 else "Non-Smoker"
            _add("SMOKING", lbl, (df_raw["SMOKING"] == sv).values)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════
# ██  CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════
def _badge(v, fmt=".1%"):
    pct = v * 100
    if   pct >= 90: return f'<span class="pill pill-green">{v:{fmt}}</span>'
    elif pct >= 75: return f'<span class="pill pill-amber">{v:{fmt}}</span>'
    else:           return f'<span class="pill pill-red">{v:{fmt}}</span>'

def _layout(**kw):
    d = {**PLOTLY_LAYOUT}
    d.update(kw)
    return d

def confusion_fig(cm, class_names, title, height=320):
    quad = {(0,0):"TN", (0,1):"FP", (1,0):"FN", (1,1):"TP"}
    txt  = [[f"{cm[i,j]}<br><sup>{quad.get((i,j),'')}</sup>"
             for j in range(cm.shape[1])]
            for i in range(cm.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Pred {c}" for c in class_names],
        y=[f"True {c}" for c in class_names],
        text=txt, texttemplate="%{text}",
        colorscale=[[0,"#EFF6FF"],[0.45,"#93C5FD"],[1,"#1D4ED8"]],
        showscale=False,
        hoverongaps=False,
        textfont=dict(size=18, color="#0F172A"),
    ))
    fig.update_layout(**_layout(
        title=dict(text=title, font=dict(size=13, color=C["text"])),
        xaxis=dict(title="Predicted", tickfont=dict(size=11)),
        yaxis=dict(title="Actual",    tickfont=dict(size=11), autorange="reversed"),
        height=height, margin=dict(l=60,r=10,t=50,b=50),
    ))
    return fig

def feat_imp_fig(imp, features, title, color=C["primary"], height=380):
    idx   = np.argsort(imp)[-12:]
    feats = np.array(features)[idx]
    vals  = imp[idx]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker=dict(
            color=vals,
            colorscale=[[0, "#DBEAFE"], [1, color]],
            showscale=False,
            line=dict(width=0),
        ),
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color=C["slate"]),
    ))
    fig.update_layout(**_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="Importance", gridcolor=C["grid"], zeroline=False, range=[0, vals.max()*1.25]),
        yaxis=dict(tickfont=dict(size=11)),
        height=height, margin=dict(l=10,r=60,t=50,b=30),
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════
# ██  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 18px;'>
      <div style='font-size:36px;'>🫁</div>
      <div style='font-size:15px;font-weight:800;color:#0F172A;letter-spacing:-.3px;'>
        Lung Cancer ML
      </div>
      <div style='font-size:11px;color:#94A3B8;margin-top:2px;'>
        Detection Intelligence Dashboard
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Upload
    st.markdown('<div class="sidebar-header">📂 Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (or use default)", type="csv",
                                 label_visibility="collapsed")

    st.markdown("---")

    # Tree models
    st.markdown('<div class="sidebar-header">🌳 Tree Classifiers</div>', unsafe_allow_html=True)
    dt_depth = st.slider("Decision Tree — max depth",  2, 15, 6)
    rf_trees = st.slider("Random Forest — n_estimators", 50, 500, 150, 50)
    gb_trees = st.slider("Gradient Boost — n_estimators", 50, 300, 150, 50)

    st.markdown("---")

    # ARM
    st.markdown('<div class="sidebar-header">🔗 Association Rules</div>', unsafe_allow_html=True)
    min_sup  = st.slider("Min Support",    0.10, 0.80, 0.30, 0.05)
    min_conf = st.slider("Min Confidence", 0.30, 1.00, 0.60, 0.05)
    min_lift = st.slider("Min Lift",       1.00, 5.00, 1.00, 0.10)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#94A3B8;line-height:1.6;'>
      Built with <b>Streamlit</b> · <b>scikit-learn</b> · <b>mlxtend</b> · <b>Plotly</b><br>
      Dataset: Lung Cancer Survey (309 records)
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# ██  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
with st.spinner("Loading and preprocessing data…"):
    try:
        if uploaded:
            csv_bytes = uploaded.read()
        else:
            with open("survey_lung_cancer.csv", "rb") as fh:
                csv_bytes = fh.read()
        P = build_pipeline(csv_bytes)
    except FileNotFoundError:
        st.error("⚠️  `survey_lung_cancer.csv` not found. Upload it via the sidebar.")
        st.stop()

with st.spinner("Training classifiers…"):
    CLF = run_classifiers(
        P["X_tr"], P["X_te"], P["y_tr"], P["y_te"],
        dt_depth, rf_trees, gb_trees, P["features"]
    )

LABEL      = P["LABEL"]
CLASS_NAMES = P["le_map"][LABEL].classes_ if LABEL in P["le_map"] else ["0","1"]


# ═══════════════════════════════════════════════════════════════════════
# ██  HERO BANNER
# ═══════════════════════════════════════════════════════════════════════
best_model_name = max(CLF, key=lambda k: CLF[k]["test_acc"])
best_acc        = CLF[best_model_name]["test_acc"]
best_f1         = CLF[best_model_name]["f1"]

st.markdown(f"""
<div class="hero">
  <div class="hero-title">🫁 Lung Cancer Detection — ML Dashboard</div>
  <div class="hero-sub">
    In-depth classification, regression, association mining &amp; bias analysis
    on {P['basic']['shape'][0]} patient survey records
  </div>
  <div style='margin-top:12px;'>
    <span class="hero-badge">309 Patients</span>
    <span class="hero-badge">{len(P['features'])} Features</span>
    <span class="hero-badge">7 Models Benchmarked</span>
    <span class="hero-badge">Best: {best_model_name} · {best_acc:.1%}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top KPI row ──────────────────────────────────────────────────────
yes_n   = int((P["df_raw"][LABEL] == "YES").sum())
no_n    = int((P["df_raw"][LABEL] == "NO").sum())
pos_pct = yes_n / P["basic"]["shape"][0]

c1,c2,c3,c4,c5,c6 = st.columns(6)
KPIS = [
    (c1, "blue",   "Total Records",    P['basic']['shape'][0],   "patients"),
    (c2, "red",    "Cancer Positive",  f"{yes_n} ({pos_pct:.0%})","LUNG_CANCER=YES"),
    (c3, "green",  "Cancer Negative",  no_n,                     "LUNG_CANCER=NO"),
    (c4, "violet", "Features Used",    len(P['features']),        "predictors"),
    (c5, "amber",  "Best Test Acc",    f"{best_acc:.1%}",         best_model_name),
    (c6, "cyan",   "Best F1 Score",    f"{best_f1:.1%}",          "weighted avg"),
]
for col, clr, lbl, val, sub in KPIS:
    col.markdown(f"""
    <div class="kpi {clr}">
      <div class="kpi-label">{lbl}</div>
      <div class="kpi-val">{val}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# ██  TABS
# ═══════════════════════════════════════════════════════════════════════
TAB_OVERVIEW, TAB_CLF, TAB_REG, TAB_ARM, TAB_BIAS = st.tabs([
    "📊  Data Overview",
    "🌳  Classification",
    "📈  Regression",
    "🔗  Association Rules",
    "⚖️   Bias Detection",
])


# ╔══════════════════════════════════════════════════════════════════════
# ║  TAB 0 – DATA OVERVIEW  (Steps 2 – 4)
# ╚══════════════════════════════════════════════════════════════════════
with TAB_OVERVIEW:

    # ── Step 2 explanation ───────────────────────────────────────────
    st.markdown("""
    <div class="info-box">
      <b>STEP 2 – Basic Data Check:</b> Shape, data types, null counts, and descriptive
      statistics are examined before any modelling. Clean data is the foundation of
      reliable predictions.
    </div>""", unsafe_allow_html=True)

    c_shape, c_null, c_dist = st.columns([1,1,1.2])

    with c_shape:
        st.markdown('<div class="sec-title"><span class="step-label">2</span>Shape & Types</div>',
                    unsafe_allow_html=True)
        info_rows = [{"Column": k, "dtype": v,
                      "Nulls": P["basic"]["null_before"][k]}
                     for k, v in P["basic"]["dtypes"].items()]
        info_df = pd.DataFrame(info_rows)
        info_df["Status"] = info_df["Nulls"].apply(
            lambda n: "✅ Clean" if n == 0 else f"⚠️ {n} missing"
        )
        st.dataframe(info_df, use_container_width=True, height=380, hide_index=True)

    with c_null:
        st.markdown('<div class="sec-title"><span class="step-label">3</span>Null Handling</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="ok-box">
          All columns are clean in this dataset.<br>
          Rule applied: continuous → <b>mean</b>; categorical → <b>mode</b>.
        </div>""", unsafe_allow_html=True)

        null_fig = go.Figure(go.Bar(
            x=list(P["basic"]["null_before"].keys()),
            y=list(P["basic"]["null_before"].values()),
            marker_color=C["primary"],
            text=list(P["basic"]["null_before"].values()),
            textposition="outside",
        ))
        null_fig.update_layout(**_layout(
            title=dict(text="Null Count per Column", font=dict(size=13)),
            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
            yaxis=dict(gridcolor=C["grid"]),
            height=300,
        ))
        st.plotly_chart(null_fig, use_container_width=True)

    with c_dist:
        st.markdown('<div class="sec-title">Label Distribution</div>', unsafe_allow_html=True)
        vc  = P["df_raw"][LABEL].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=vc.index, values=vc.values,
            hole=0.62,
            marker=dict(colors=[C["yes_clr"], C["no_clr"]],
                        line=dict(color="#FFFFFF", width=3)),
            textinfo="label+percent",
            textfont=dict(size=13, family="Plus Jakarta Sans"),
        ))
        fig_donut.add_annotation(
            text=f"<b>{vc.sum()}</b><br>patients",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=C["text"], family="Plus Jakarta Sans"),
        )
        fig_donut.update_layout(**_layout(
            title=dict(text="LUNG_CANCER Class Split", font=dict(size=13)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            height=300,
        ))
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Descriptive stats ────────────────────────────────────────────
    st.markdown('<div class="sec-title">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(P["basic"]["describe"].round(3), use_container_width=True, height=230)

    # ── Step 4: Label Encoding ────────────────────────────────────────
    st.markdown("""
    <div class="info-box">
      <b>STEP 4 – Label Encoding:</b> Categorical columns (GENDER, LUNG_CANCER) are
      integer-encoded using <code>sklearn.LabelEncoder</code>.
      The mapping is stored below so predictions can be decoded back to original labels.
    </div>""", unsafe_allow_html=True)

    col_map, col_prev = st.columns([1, 2])

    with col_map:
        st.markdown('<div class="sec-title">Encoding Map</div>', unsafe_allow_html=True)
        st.dataframe(P["map_df"], use_container_width=True, height=220, hide_index=True)
        csv_out = P["df_enc"].to_csv(index=False).encode()
        st.download_button("⬇️  Download Encoded CSV", csv_out,
                           "lung_cancer_encoded.csv", "text/csv",
                           use_container_width=True)

    with col_prev:
        st.markdown('<div class="sec-title">Encoded Dataset (first 20 rows)</div>',
                    unsafe_allow_html=True)
        st.dataframe(P["df_enc"].head(20), use_container_width=True, height=220)

    # ── Correlation with LUNG_CANCER ──────────────────────────────────
    st.markdown('<div class="sec-title">Feature Correlation with LUNG_CANCER</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      Point-biserial correlation measures the linear relationship between each feature
      and the binary label. Features with higher absolute correlation are stronger
      individual predictors.
    </div>""", unsafe_allow_html=True)

    corr_s = P["corr"]
    fig_corr = go.Figure(go.Bar(
        x=corr_s.values, y=corr_s.index,
        orientation="h",
        marker=dict(
            color=corr_s.values,
            colorscale=[[0,"#DBEAFE"],[1,C["primary"]]],
            showscale=True,
            colorbar=dict(title="|r|", thickness=12),
        ),
        text=[f"{v:.3f}" for v in corr_s.values],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig_corr.update_layout(**_layout(
        title=dict(text="Absolute Correlation |r| with LUNG_CANCER", font=dict(size=14)),
        xaxis=dict(title="|Correlation|", range=[0, corr_s.max()*1.25],
                   gridcolor=C["grid"]),
        yaxis=dict(tickfont=dict(size=11)),
        height=450,
    ))
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Chi-square significance ───────────────────────────────────────
    st.markdown('<div class="sec-title">Chi-Square Feature Significance Test</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      Chi-square tests whether each feature is <b>statistically independent</b> from
      LUNG_CANCER. p &lt; 0.05 rejects independence — the feature carries predictive
      signal.
    </div>""", unsafe_allow_html=True)

    chi_html = """
    <table class="stbl">
      <thead><tr>
        <th>#</th><th>Feature</th><th>Chi² Statistic</th><th>p-value</th><th>Significant?</th>
      </tr></thead><tbody>"""
    for i, row in P["chi2_df"].iterrows():
        sig_badge = '<span class="pill pill-green">✅ Yes</span>' if "Yes" in row["Significant"] \
                    else '<span class="pill pill-red">❌ No</span>'
        chi_html += f"""
      <tr>
        <td>{i+1}</td>
        <td><b>{row['Feature']}</b></td>
        <td>{row['Chi2']}</td>
        <td>{row['p-value']}</td>
        <td>{sig_badge}</td>
      </tr>"""
    chi_html += "</tbody></table>"
    st.markdown(chi_html, unsafe_allow_html=True)

    # ── Age distribution ──────────────────────────────────────────────
    st.markdown('<div class="sec-title">Age Distribution by Cancer Status</div>',
                unsafe_allow_html=True)
    fig_age = go.Figure()
    for val, clr, lbl in [("YES", C["yes_clr"], "Cancer YES"),
                           ("NO",  C["no_clr"],  "Cancer NO")]:
        subset = P["df_raw"][P["df_raw"][LABEL] == val]["AGE"]
        fig_age.add_trace(go.Histogram(
            x=subset, name=lbl, nbinsx=20,
            marker_color=clr, opacity=0.75,
        ))
    fig_age.update_layout(**_layout(
        barmode="overlay",
        title=dict(text="Age Distribution — YES vs NO", font=dict(size=14)),
        xaxis=dict(title="Age", gridcolor=C["grid"]),
        yaxis=dict(title="Count", gridcolor=C["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=320,
    ))
    st.plotly_chart(fig_age, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════
# ║  TAB 1 – CLASSIFICATION  (Steps 7 – 10)
# ╚══════════════════════════════════════════════════════════════════════
with TAB_CLF:

    # ── Step 6 info ──────────────────────────────────────────────────
    st.markdown("""
    <div class="info-box">
      <b>STEPS 5–6:</b> Features &amp; label split complete.
      Train set: <b>{}</b> samples | Test set: <b>{}</b> samples | 80:20 stratified split.
    </div>""".format(len(P["X_tr"]), len(P["X_te"])), unsafe_allow_html=True)

    # ── STEP 8: Accuracy table ────────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="step-label">8</span>Model Performance — All Algorithms</div>',
                unsafe_allow_html=True)

    rows = []
    for nm, r in CLF.items():
        rows.append({
            "Model"        : nm,
            "train_acc"    : r["train_acc"],
            "test_acc"     : r["test_acc"],
            "precision"    : r["precision"],
            "recall"       : r["recall"],
            "f1"           : r["f1"],
            "auc"          : r["auc"],
            "cv_mean"      : r["cv_mean"],
        })
    acc_df = pd.DataFrame(rows)

    tbl = """
    <table class="stbl">
      <thead><tr>
        <th>Model</th><th>Train Acc</th><th>Test Acc</th>
        <th>Precision</th><th>Recall</th><th>F1 Score</th>
        <th>ROC-AUC</th><th>5-Fold CV</th>
      </tr></thead><tbody>"""
    for _, r in acc_df.iterrows():
        auc_str = _badge(r["auc"]) if r["auc"] else "—"
        tbl += f"""
      <tr>
        <td><b style='color:{MODEL_COLOURS.get(r['Model'], C['text'])};'>{r['Model']}</b></td>
        <td>{_badge(r['train_acc'])}</td>
        <td>{_badge(r['test_acc'])}</td>
        <td>{r['precision']*100:.1f}%</td>
        <td>{r['recall']*100:.1f}%</td>
        <td>{r['f1']*100:.1f}%</td>
        <td>{auc_str}</td>
        <td>{r['cv_mean']*100:.1f}%</td>
      </tr>"""
    tbl += "</tbody></table>"
    st.markdown(tbl, unsafe_allow_html=True)

    # ── Grouped bar ───────────────────────────────────────────────────
    metrics   = ["test_acc", "precision", "recall", "f1"]
    m_labels  = ["Test Accuracy", "Precision", "Recall", "F1 Score"]
    m_colours = [C["primary"], C["success"], C["warning"], C["violet"]]

    fig_grp = go.Figure()
    for met, lbl, clr in zip(metrics, m_labels, m_colours):
        fig_grp.add_trace(go.Bar(
            name=lbl,
            x=acc_df["Model"],
            y=acc_df[met] * 100,
            marker_color=clr,
            text=[f"{v*100:.1f}%" for v in acc_df[met]],
            textposition="outside", textfont=dict(size=10),
        ))
    fig_grp.update_layout(**_layout(
        barmode="group",
        title=dict(text="Model Metrics Comparison", font=dict(size=14)),
        yaxis=dict(range=[0,115], gridcolor=C["grid"], title="Score (%)"),
        xaxis=dict(tickangle=-10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
    ))
    st.plotly_chart(fig_grp, use_container_width=True)

    # ── ROC Curves ────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">ROC Curves — All Models</div>', unsafe_allow_html=True)
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dot", color=C["slate"], width=1))
    for nm, r in CLF.items():
        if r["prob_te"] is not None:
            fpr, tpr, _ = roc_curve(P["y_te"], r["prob_te"])
            auc_val = r["auc"]
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{nm} (AUC={auc_val:.3f})",
                line=dict(color=MODEL_COLOURS[nm], width=2.5),
                mode="lines",
            ))
    fig_roc.update_layout(**_layout(
        title=dict(text="ROC Curves — Test Set", font=dict(size=14)),
        xaxis=dict(title="False Positive Rate", gridcolor=C["grid"], range=[0,1]),
        yaxis=dict(title="True Positive Rate",  gridcolor=C["grid"], range=[0,1.02]),
        legend=dict(x=0.55, y=0.08, bgcolor="#FFFFFF",
                    bordercolor="#E2E8F0", borderwidth=1),
        height=420,
    ))
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── STEP 9: Confusion matrices ────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="step-label">9</span>Confusion Matrices</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      <b>TP</b> = True Positive &nbsp;|&nbsp;
      <b>TN</b> = True Negative &nbsp;|&nbsp;
      <b>FP</b> = False Positive (healthy predicted as cancer) &nbsp;|&nbsp;
      <b>FN</b> = False Negative (cancer missed — most critical error)
    </div>""", unsafe_allow_html=True)

    # Row 1: tree models
    row1 = ["Decision Tree", "Random Forest", "Gradient Boosting"]
    cols_r1 = st.columns(3)
    for col_w, nm in zip(cols_r1, row1):
        with col_w:
            st.plotly_chart(
                confusion_fig(CLF[nm]["cm"], CLASS_NAMES, nm),
                use_container_width=True,
            )
    # Row 2: remaining
    row2 = ["Logistic Regression", "SVM", "KNN"]
    cols_r2 = st.columns(3)
    for col_w, nm in zip(cols_r2, row2):
        with col_w:
            st.plotly_chart(
                confusion_fig(CLF[nm]["cm"], CLASS_NAMES, nm),
                use_container_width=True,
            )

    # ── STEP 10: Feature importance ───────────────────────────────────
    st.markdown('<div class="sec-title"><span class="step-label">10</span>Feature Importances</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      Tree-based models expose internal feature importance (Gini impurity reduction).
      Higher values indicate the feature contributes more to splitting decisions
      and therefore to cancer detection.
    </div>""", unsafe_allow_html=True)

    tree_models = {k: v for k, v in CLF.items() if v["imp"] is not None}
    fi_cols = st.columns(len(tree_models))
    for col_w, (nm, r) in zip(fi_cols, tree_models.items()):
        with col_w:
            st.plotly_chart(
                feat_imp_fig(r["imp"], P["features"], nm,
                             color=MODEL_COLOURS[nm]),
                use_container_width=True,
            )

    # ── Precision-Recall curves ───────────────────────────────────────
    st.markdown('<div class="sec-title">Precision-Recall Curves</div>', unsafe_allow_html=True)
    fig_pr = go.Figure()
    for nm, r in CLF.items():
        if r["prob_te"] is not None:
            prec, rec, _ = precision_recall_curve(P["y_te"], r["prob_te"])
            ap = average_precision_score(P["y_te"], r["prob_te"])
            fig_pr.add_trace(go.Scatter(
                x=rec, y=prec,
                name=f"{nm} (AP={ap:.3f})",
                line=dict(color=MODEL_COLOURS[nm], width=2.5),
                mode="lines",
            ))
    fig_pr.update_layout(**_layout(
        title=dict(text="Precision-Recall Curves — Test Set", font=dict(size=14)),
        xaxis=dict(title="Recall",    gridcolor=C["grid"]),
        yaxis=dict(title="Precision", gridcolor=C["grid"]),
        legend=dict(x=0.01, y=0.01, bgcolor="#FFFFFF",
                    bordercolor="#E2E8F0", borderwidth=1),
        height=400,
    ))
    st.plotly_chart(fig_pr, use_container_width=True)

    # ── Per-class report ──────────────────────────────────────────────
    st.markdown('<div class="sec-title">Detailed Classification Report</div>',
                unsafe_allow_html=True)
    sel = st.selectbox("Choose model", list(CLF.keys()), key="sel_model")
    cr  = classification_report(P["y_te"], CLF[sel]["yp_te"],
                                  target_names=CLASS_NAMES, output_dict=True)
    cr_df = pd.DataFrame(cr).T.round(3)
    st.dataframe(cr_df, use_container_width=True)

    # ── CV box plot ───────────────────────────────────────────────────
    st.markdown('<div class="sec-title">5-Fold Cross-Validation Distribution</div>',
                unsafe_allow_html=True)
    @st.cache_data(show_spinner=False)
    def _cv_scores(X_tr, y_tr, dt_d, rf_n, gb_n):
        mdls = {
            "Decision Tree"       : DecisionTreeClassifier(random_state=42, max_depth=dt_d),
            "Random Forest"       : RandomForestClassifier(n_estimators=rf_n, random_state=42),
            "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=gb_n, random_state=42),
            "Logistic Regression" : LogisticRegression(max_iter=500, random_state=42),
            "SVM"                 : SVC(probability=True, random_state=42),
        }
        return {nm: cross_val_score(m, X_tr, y_tr, cv=StratifiedKFold(5),
                                     scoring="accuracy")
                for nm, m in mdls.items()}

    cv_scores = _cv_scores(P["X_tr"], P["y_tr"], dt_depth, rf_trees, gb_trees)
    fig_cv = go.Figure()
    for nm, scores in cv_scores.items():
        fig_cv.add_trace(go.Box(
            y=scores * 100, name=nm,
            marker_color=MODEL_COLOURS[nm],
            boxmean="sd", line=dict(width=2),
        ))
    fig_cv.update_layout(**_layout(
        title=dict(text="5-Fold CV Accuracy Distribution", font=dict(size=14)),
        yaxis=dict(title="Accuracy (%)", gridcolor=C["grid"]),
        height=360,
    ))
    st.plotly_chart(fig_cv, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════
# ║  TAB 2 – REGRESSION  (logistic family)
# ╚══════════════════════════════════════════════════════════════════════
with TAB_REG:
    st.markdown("""
    <div class="info-box">
      <b>Regression-family classifiers</b> — Logistic Regression, Ridge Classifier, and
      SVM operate on linear decision boundaries and probabilistic frameworks rather than
      tree splits. They complement tree models and can outperform them on smaller datasets.
    </div>""", unsafe_allow_html=True)

    reg_names = ["Logistic Regression", "SVM", "KNN", "Naive Bayes"]

    # ── Metrics table ─────────────────────────────────────────────────
    st.markdown('<div class="sec-title">Regression-family Model Metrics</div>',
                unsafe_allow_html=True)

    rtbl = """
    <table class="stbl">
      <thead><tr>
        <th>Model</th><th>Train Acc</th><th>Test Acc</th>
        <th>Precision</th><th>Recall</th><th>F1 Score</th><th>ROC-AUC</th>
      </tr></thead><tbody>"""
    for nm in reg_names:
        r = CLF[nm]
        auc_str = f"{r['auc']:.3f}" if r['auc'] else "—"
        rtbl += f"""
      <tr>
        <td><b style='color:{MODEL_COLOURS[nm]};'>{nm}</b></td>
        <td>{_badge(r['train_acc'])}</td>
        <td>{_badge(r['test_acc'])}</td>
        <td>{r['precision']*100:.1f}%</td>
        <td>{r['recall']*100:.1f}%</td>
        <td>{r['f1']*100:.1f}%</td>
        <td>{auc_str}</td>
      </tr>"""
    rtbl += "</tbody></table>"
    st.markdown(rtbl, unsafe_allow_html=True)

    # ── Radar chart ───────────────────────────────────────────────────
    st.markdown('<div class="sec-title">Radar — Performance Across All Metrics</div>',
                unsafe_allow_html=True)

    all_names = list(CLF.keys())
    cats = ["Test Acc", "Precision", "Recall", "F1", "CV Mean"]
    fig_radar = go.Figure()
    for nm in all_names:
        r = CLF[nm]
        vals = [r["test_acc"], r["precision"], r["recall"], r["f1"], r["cv_mean"]]
        vals += [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", name=nm,
            line=dict(color=MODEL_COLOURS[nm], width=2),
            fillcolor=MODEL_COLOURS[nm],
            opacity=0.15,
        ))
    fig_radar.update_layout(**_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0.5, 1.05],
                            tickformat=".0%", gridcolor=C["grid"]),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        title=dict(text="All Models — Radar (test set)", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        height=500,
    ))
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Confusion matrices ────────────────────────────────────────────
    st.markdown('<div class="sec-title">Confusion Matrices — Regression Models</div>',
                unsafe_allow_html=True)
    rcols = st.columns(4)
    for col_w, nm in zip(rcols, reg_names):
        with col_w:
            st.plotly_chart(
                confusion_fig(CLF[nm]["cm"], CLASS_NAMES, nm, height=260),
                use_container_width=True,
            )

    # ── All-model test-acc rank ───────────────────────────────────────
    st.markdown('<div class="sec-title">All Models — Test Accuracy Ranking</div>',
                unsafe_allow_html=True)

    rank_df = acc_df.sort_values("test_acc", ascending=True)
    fig_rank = go.Figure(go.Bar(
        x=rank_df["test_acc"] * 100,
        y=rank_df["Model"],
        orientation="h",
        marker=dict(
            color=[MODEL_COLOURS[nm] for nm in rank_df["Model"]],
            line=dict(width=0),
        ),
        text=[f"{v*100:.1f}%" for v in rank_df["test_acc"]],
        textposition="outside",
        textfont=dict(size=12),
    ))
    fig_rank.update_layout(**_layout(
        title=dict(text="Test Accuracy Ranking", font=dict(size=14)),
        xaxis=dict(title="Test Accuracy (%)", range=[0, 115], gridcolor=C["grid"]),
        yaxis=dict(tickfont=dict(size=12)),
        height=380,
    ))
    st.plotly_chart(fig_rank, use_container_width=True)

    # ── Logistic Regression coefficients ──────────────────────────────
    st.markdown('<div class="sec-title">Logistic Regression — Feature Coefficients</div>',
                unsafe_allow_html=True)
    lr  = CLF["Logistic Regression"]["model"]
    coef = lr.coef_[0]
    coef_df = pd.DataFrame({
        "Feature": P["features"], "Coefficient": coef
    }).sort_values("Coefficient", key=abs, ascending=False)

    fig_coef = go.Figure(go.Bar(
        x=coef_df["Coefficient"],
        y=coef_df["Feature"],
        orientation="h",
        marker=dict(
            color=coef_df["Coefficient"],
            colorscale=[[0, C["yes_clr"]], [0.5, "#F8FAFC"], [1, C["primary"]]],
            cmid=0, showscale=True,
        ),
    ))
    fig_coef.update_layout(**_layout(
        title=dict(text="Logistic Regression Coefficients (signed)", font=dict(size=14)),
        xaxis=dict(title="Coefficient", gridcolor=C["grid"], zeroline=True,
                   zerolinecolor=C["slate"]),
        yaxis=dict(tickfont=dict(size=11)),
        height=420,
    ))
    st.plotly_chart(fig_coef, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════
# ║  TAB 3 – ASSOCIATION RULE MINING
# ╚══════════════════════════════════════════════════════════════════════
with TAB_ARM:
    st.markdown("""
    <div class="info-box">
      <b>Association Rule Mining</b> uses the <b>Apriori</b> algorithm to find
      co-occurring symptom patterns. Rules with high <b>Lift</b> reveal which symptom
      combinations strongly co-exist with lung cancer diagnosis.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Running Apriori…"):
        freq_items, rules = run_arm(P["df_enc"], min_sup, min_conf, min_lift)

    if rules.empty:
        st.markdown("""
        <div class="warn-box">
          No rules found at the current thresholds. Lower Min Support or Min Confidence
          in the sidebar.
        </div>""", unsafe_allow_html=True)
    else:
        # KPI row
        ka, kb, kc, kd, ke = st.columns(5)
        arm_kpis = [
            (ka, "blue",   "Rules Found",    len(rules),                   "passing all filters"),
            (kb, "green",  "Avg Confidence", f"{rules['confidence'].mean():.1%}", "mean rule confidence"),
            (kc, "amber",  "Max Lift",        f"{rules['lift'].max():.2f}", "highest lift"),
            (kd, "violet", "Avg Support",     f"{rules['support'].mean():.1%}","mean item-set freq"),
            (ke, "cyan",   "Freq Item-sets",  len(freq_items),              "above min support"),
        ]
        for col_w, clr, lbl, val, sub in arm_kpis:
            col_w.markdown(f"""
            <div class="kpi {clr}">
              <div class="kpi-label">{lbl}</div>
              <div class="kpi-val">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Full rules table ─────────────────────────────────────────
        st.markdown('<div class="sec-title">All Association Rules</div>',
                    unsafe_allow_html=True)

        disp = rules[["antecedents","consequents",
                       "support","confidence","lift","leverage","conviction"]].copy()
        disp["support"]    = (disp["support"]    * 100).round(2).astype(str) + "%"
        disp["confidence"] = (disp["confidence"] * 100).round(2).astype(str) + "%"
        disp["lift"]       = disp["lift"].round(3)
        disp["leverage"]   = disp["leverage"].round(4)
        disp["conviction"] = disp["conviction"].round(3)
        disp.index = disp.index + 1
        st.dataframe(disp, use_container_width=True, height=360)

        # ── Top 20 Lift bar ───────────────────────────────────────────
        st.markdown('<div class="sec-title">Top 20 Rules — Lift Score</div>',
                    unsafe_allow_html=True)
        top20 = rules.head(20).copy()
        top20["Rule"] = top20["antecedents"] + " → " + top20["consequents"]

        fig_lift_bar = go.Figure(go.Bar(
            x=top20["lift"], y=top20["Rule"],
            orientation="h",
            marker=dict(
                color=top20["lift"],
                colorscale=[[0,"#DBEAFE"],[1,C["primary"]]],
                showscale=True,
                colorbar=dict(title="Lift", thickness=12),
            ),
            text=[f"Lift {v:.2f} | Conf {c:.0%}"
                  for v, c in zip(top20["lift"], top20["confidence"])],
            textposition="outside",
            textfont=dict(size=10),
        ))
        fig_lift_bar.update_layout(**_layout(
            title=dict(text="Top 20 Rules by Lift", font=dict(size=14)),
            xaxis=dict(title="Lift", gridcolor=C["grid"]),
            height=600, margin=dict(l=10, r=80, t=50, b=30),
        ))
        st.plotly_chart(fig_lift_bar, use_container_width=True)

        # ── Support vs Confidence scatter ─────────────────────────────
        st.markdown('<div class="sec-title">Support vs Confidence — coloured by Lift</div>',
                    unsafe_allow_html=True)
        fig_scatter = go.Figure(go.Scatter(
            x=rules["support"],
            y=rules["confidence"],
            mode="markers",
            marker=dict(
                size=rules["lift"] * 5,
                color=rules["lift"],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Lift", thickness=12),
                opacity=0.8,
                line=dict(width=1, color="#FFFFFF"),
            ),
            text=[f"{a} → {c}<br>Lift: {l:.2f}"
                  for a, c, l in zip(rules["antecedents"],
                                      rules["consequents"],
                                      rules["lift"])],
            hovertemplate="%{text}<extra></extra>",
        ))
        fig_scatter.update_layout(**_layout(
            title=dict(text="Support vs Confidence (bubble = Lift)", font=dict(size=14)),
            xaxis=dict(title="Support", gridcolor=C["grid"]),
            yaxis=dict(title="Confidence", gridcolor=C["grid"]),
            height=440,
        ))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ── LUNG_CANCER-specific rules ────────────────────────────────
        lc_rules = rules[rules["consequents"].str.contains("LUNG_CANCER", na=False)].copy()
        if not lc_rules.empty:
            st.markdown('<div class="sec-title">🫁 Rules → LUNG_CANCER_YES</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div class="danger-box">
              These rules reveal which symptom combinations directly predict a positive
              lung cancer diagnosis. High-lift rules are the most actionable clinical signals.
            </div>""", unsafe_allow_html=True)

            disp_lc = lc_rules[["antecedents","consequents",
                                  "support","confidence","lift"]].copy()
            disp_lc["support"]    = (disp_lc["support"]    * 100).round(2).astype(str) + "%"
            disp_lc["confidence"] = (disp_lc["confidence"] * 100).round(2).astype(str) + "%"
            disp_lc["lift"]       = disp_lc["lift"].round(3)
            st.dataframe(disp_lc, use_container_width=True)
        else:
            st.markdown("""
            <div class="warn-box">
              No rules with LUNG_CANCER_YES as consequent at current settings.
              Lower Confidence threshold to reveal them.
            </div>""", unsafe_allow_html=True)

        # ── Metric explainer ─────────────────────────────────────────
        with st.expander("📖 How to interpret ARM metrics"):
            st.markdown("""
| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Support** | P(A ∪ B) | How often itemset appears in data (frequency) |
| **Confidence** | P(B\|A) | If A present, probability B also present |
| **Lift** | Confidence / P(B) | >1 = positive association; 1 = independent; <1 = negative |
| **Leverage** | P(A∪B) − P(A)·P(B) | Excess co-occurrence over independence; 0 = no association |
| **Conviction** | (1−P(B))/(1−Conf) | ∞ = perfect rule; higher = stronger implication |

**Rule of thumb for clinical screening:**
- Lift > 2 → strong co-occurrence worth investigating
- Confidence > 80% → reliable enough to flag patients
- Look for high-lift rules leading to LUNG_CANCER_YES
            """)


# ╔══════════════════════════════════════════════════════════════════════
# ║  TAB 4 – BIAS DETECTION
# ╚══════════════════════════════════════════════════════════════════════
with TAB_BIAS:
    st.markdown("""
    <div class="danger-box">
      <b>⚖️ Why bias detection matters in medical AI:</b> A model that performs well
      overall can still systematically under-diagnose cancer in specific demographic
      groups (gender, age, smoking status), leading to life-threatening missed diagnoses.
      This tab quantifies that risk.
    </div>""", unsafe_allow_html=True)

    best_r     = CLF[best_model_name]
    bias_df    = run_bias(P["df_raw"], P["df_enc"], best_r["model"], LABEL, P["features"])
    overall_acc = best_r["test_acc"]

    st.markdown(
        f'<div class="ok-box">Best model used for bias evaluation: '
        f'<b>{best_model_name}</b> — Overall Test Accuracy: '
        f'<b>{overall_acc:.1%}</b></div>',
        unsafe_allow_html=True
    )

    # ── Bias summary table ────────────────────────────────────────────
    st.markdown('<div class="sec-title">Group-level Performance Gaps</div>',
                unsafe_allow_html=True)

    if not bias_df.empty:
        btbl = """
        <table class="stbl">
          <thead><tr>
            <th>Group Dimension</th><th>Group Value</th><th>N</th>
            <th>Accuracy</th><th>Positive Rate</th>
            <th>TPR (Recall)</th><th>Acc Gap vs Overall</th>
          </tr></thead><tbody>"""
        for _, row in bias_df.iterrows():
            gap     = row["Accuracy"] - overall_acc
            gap_str = f"+{gap*100:.1f}%" if gap >= 0 else f"{gap*100:.1f}%"
            gap_clr = "#065F46" if gap >= 0 else "#991B1B"
            btbl += f"""
          <tr>
            <td><b>{row['Group Column']}</b></td>
            <td>{row['Group Value']}</td>
            <td>{row['N']}</td>
            <td>{_badge(row['Accuracy'])}</td>
            <td>{row['Positive Rate']*100:.1f}%</td>
            <td>{row['TPR (Recall)']*100:.1f}%</td>
            <td style='color:{gap_clr};font-weight:700;'>{gap_str}</td>
          </tr>"""
        btbl += "</tbody></table>"
        st.markdown(btbl, unsafe_allow_html=True)
    else:
        st.warning("Not enough group data to compute bias metrics.")

    # ── Accuracy gap bar ──────────────────────────────────────────────
    if not bias_df.empty:
        st.markdown('<div class="sec-title">Accuracy by Demographic Group</div>',
                    unsafe_allow_html=True)
        for group_col in bias_df["Group Column"].unique():
            sub = bias_df[bias_df["Group Column"] == group_col].copy()
            sub["Acc %"] = sub["Accuracy"] * 100
            sub["PR %"]  = sub["Positive Rate"] * 100

            fig_g = go.Figure()
            fig_g.add_trace(go.Bar(
                x=sub["Group Value"], y=sub["Acc %"],
                name="Model Accuracy",
                marker_color=C["primary"],
                text=[f"{v:.1f}%" for v in sub["Acc %"]],
                textposition="outside",
            ))
            fig_g.add_trace(go.Bar(
                x=sub["Group Value"], y=sub["PR %"],
                name="Positive Rate",
                marker_color=C["warning"],
                text=[f"{v:.1f}%" for v in sub["PR %"]],
                textposition="outside",
            ))
            fig_g.add_hline(
                y=overall_acc * 100,
                line_dash="dash", line_color=C["danger"], line_width=1.5,
                annotation_text=f"Overall {overall_acc:.1%}",
                annotation_position="top right",
                annotation_font=dict(color=C["danger"], size=11),
            )
            fig_g.update_layout(**_layout(
                barmode="group",
                title=dict(text=f"Bias — {group_col}", font=dict(size=14)),
                yaxis=dict(range=[0,115], gridcolor=C["grid"], title="% Score"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=360,
            ))
            st.plotly_chart(fig_g, use_container_width=True)

    # ── Disparate Impact Ratio ────────────────────────────────────────
    st.markdown('<div class="sec-title">📐 Disparate Impact Ratio (DIR)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      <b>DIR = Positive Rate (disadvantaged group) / Positive Rate (advantaged group)</b><br>
      Under the US EEOC "4/5ths rule", a DIR &lt; 0.80 indicates potential discriminatory impact.
      In medical AI, a low DIR means one group is being diagnosed at significantly lower rates.
    </div>""", unsafe_allow_html=True)

    gender_sub = bias_df[bias_df["Group Column"] == "GENDER"] if not bias_df.empty else pd.DataFrame()
    if len(gender_sub) >= 2:
        g_sorted = gender_sub.sort_values("Positive Rate", ascending=False)
        maj_rate = g_sorted.iloc[0]["Positive Rate"]
        min_rate = g_sorted.iloc[-1]["Positive Rate"]
        dir_val  = min_rate / maj_rate if maj_rate > 0 else 1.0

        d1, d2, d3, d4 = st.columns(4)
        d1.markdown(f"""<div class="kpi blue">
          <div class="kpi-label">Majority +ve Rate</div>
          <div class="kpi-val">{maj_rate*100:.1f}%</div>
          <div class="kpi-sub">{g_sorted.iloc[0]['Group Value']}</div>
        </div>""", unsafe_allow_html=True)
        d2.markdown(f"""<div class="kpi amber">
          <div class="kpi-label">Minority +ve Rate</div>
          <div class="kpi-val">{min_rate*100:.1f}%</div>
          <div class="kpi-sub">{g_sorted.iloc[-1]['Group Value']}</div>
        </div>""", unsafe_allow_html=True)
        dir_clr   = "green" if dir_val >= 0.80 else "red"
        dir_label = "✅ Fair (≥ 0.80)" if dir_val >= 0.80 else "⚠️ Biased (< 0.80)"
        d3.markdown(f"""<div class="kpi {dir_clr}">
          <div class="kpi-label">Disparate Impact Ratio</div>
          <div class="kpi-val">{dir_val:.3f}</div>
          <div class="kpi-sub">{dir_label}</div>
        </div>""", unsafe_allow_html=True)
        tpr_gap = abs(gender_sub["TPR (Recall)"].max() - gender_sub["TPR (Recall)"].min())
        d4.markdown(f"""<div class="kpi violet">
          <div class="kpi-label">TPR Gap (Gender)</div>
          <div class="kpi-val">{tpr_gap*100:.1f}%</div>
          <div class="kpi-sub">Equal Opportunity gap</div>
        </div>""", unsafe_allow_html=True)

    # ── Distribution bias ─────────────────────────────────────────────
    st.markdown('<div class="sec-title">Dataset Representation Bias</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="warn-box">
      Representation bias occurs when certain groups are under-represented in training data.
      The model learns less about under-represented groups and may perform poorly for them.
    </div>""", unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if "GENDER" in P["df_raw"].columns:
            gvc = P["df_raw"].groupby(["GENDER", LABEL]).size().reset_index(name="Count")
            fig_gvc = px.bar(gvc, x="GENDER", y="Count", color=LABEL,
                             barmode="group", template="simple_white",
                             color_discrete_map={"YES": C["yes_clr"], "NO": C["no_clr"]},
                             title="Gender × Cancer Status")
            fig_gvc.update_layout(**_layout(title=dict(text="Gender × Cancer Status",
                                                        font=dict(size=13)), height=320))
            st.plotly_chart(fig_gvc, use_container_width=True)

    with col_b2:
        if "AGE" in P["df_raw"].columns:
            df_tmp2 = P["df_raw"].copy()
            df_tmp2["Age Group"] = pd.cut(df_tmp2["AGE"], bins=[0,40,55,70,200],
                                           labels=["<40","40-55","56-70",">70"])
            agvc = df_tmp2.groupby(["Age Group", LABEL]).size().reset_index(name="Count")
            fig_avc = px.bar(agvc, x="Age Group", y="Count", color=LABEL,
                             barmode="group", template="simple_white",
                             color_discrete_map={"YES": C["yes_clr"], "NO": C["no_clr"]},
                             title="Age Group × Cancer Status")
            fig_avc.update_layout(**_layout(title=dict(text="Age Group × Cancer Status",
                                                         font=dict(size=13)), height=320))
            st.plotly_chart(fig_avc, use_container_width=True)

    # ── Class imbalance ───────────────────────────────────────────────
    st.markdown('<div class="sec-title">Class Imbalance Analysis</div>',
                unsafe_allow_html=True)
    vc2 = P["df_raw"][LABEL].value_counts()
    imbalance_ratio = vc2.min() / vc2.max()

    ci1, ci2, ci3 = st.columns(3)
    ci1.markdown(f"""<div class="kpi red">
      <div class="kpi-label">Imbalance Ratio</div>
      <div class="kpi-val">{imbalance_ratio:.2f}</div>
      <div class="kpi-sub">minority / majority</div>
    </div>""", unsafe_allow_html=True)
    ci2.markdown(f"""<div class="kpi green">
      <div class="kpi-label">Majority Class</div>
      <div class="kpi-val">{vc2.index[0]} ({vc2.iloc[0]})</div>
      <div class="kpi-sub">{vc2.iloc[0]/vc2.sum():.0%} of data</div>
    </div>""", unsafe_allow_html=True)
    ci3.markdown(f"""<div class="kpi amber">
      <div class="kpi-label">Minority Class</div>
      <div class="kpi-val">{vc2.index[-1]} ({vc2.iloc[-1]})</div>
      <div class="kpi-sub">{vc2.iloc[-1]/vc2.sum():.0%} of data</div>
    </div>""", unsafe_allow_html=True)

    imbal_note = "mild" if imbalance_ratio > 0.5 else "moderate" if imbalance_ratio > 0.2 else "severe"
    st.markdown(f"""
    <div class="{'ok-box' if imbalance_ratio > 0.5 else 'warn-box'}">
      Class imbalance is <b>{imbal_note}</b> (ratio {imbalance_ratio:.2f}).
      {'No resampling needed.' if imbalance_ratio > 0.5 else
       'Consider SMOTE oversampling or class_weight="balanced" to reduce minority-class bias.'}
    </div>""", unsafe_allow_html=True)

    # ── Mitigation playbook ───────────────────────────────────────────
    st.markdown('<div class="sec-title">💡 Bias Mitigation Playbook</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="ok-box">
    <ol style='margin:0;padding-left:20px;line-height:2;'>
      <li><b>Re-weighting:</b> Pass <code>class_weight='balanced'</code> to tree/logistic models.</li>
      <li><b>SMOTE oversampling:</b> Use <code>imbalanced-learn</code> to synthetically oversample the minority class.</li>
      <li><b>Fairness-aware training:</b> Use <code>Fairlearn</code> or <code>AIF360</code> with demographic-parity constraints.</li>
      <li><b>Equalised Odds:</b> Optimise for equal TPR / FPR across gender and age groups post-hoc.</li>
      <li><b>Stratified evaluation:</b> Always report group-level metrics — overall accuracy hides group-level failure.</li>
      <li><b>Data collection:</b> Ensure training data proportionally represents all demographic groups.</li>
      <li><b>Clinical review:</b> Flag high-risk, under-represented patients for mandatory clinician review.</li>
    </ol>
    </div>""", unsafe_allow_html=True)
