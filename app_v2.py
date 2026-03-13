"""
╔══════════════════════════════════════════════════════════════════════╗
║   LUNG CANCER DETECTION  –  ML Intelligence Dashboard  v2           ║
║   25+ interactive graph types across 5 tabs                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import warnings, io
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      StratifiedKFold, learning_curve)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
from mlxtend.frequent_patterns import apriori, association_rules

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Lung Cancer ML Dashboard", page_icon="🫁",
                   layout="wide", initial_sidebar_state="expanded")

# ════════════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#FAFBFF!important;font-family:'Plus Jakarta Sans',sans-serif!important;}
[data-testid="stSidebar"]{background:#FFF!important;border-right:1.5px solid #E2E8F0!important;}
[data-testid="stHeader"],[data-testid="stToolbar"]{background:transparent!important;display:block;}
[data-testid="stToolbar"]{display:none!important;}
h1,h2,h3,h4,h5,h6,[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2{font-family:'Plus Jakarta Sans',sans-serif!important;color:#0F172A!important;}
p,li,td,th{font-family:'Plus Jakarta Sans',sans-serif!important;}
code,pre{font-family:'JetBrains Mono',monospace!important;}
.hero{background:linear-gradient(135deg,#1E3A5F 0%,#2563EB 55%,#0EA5E9 100%);border-radius:20px;padding:36px 40px;margin-bottom:28px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;inset:0;background:radial-gradient(circle at 78% 50%,rgba(255,255,255,.09) 0%,transparent 58%);}
.hero-title{font-size:30px;font-weight:800;color:#FFF;letter-spacing:-.5px;margin:0 0 6px;}
.hero-sub{font-size:14px;color:rgba(255,255,255,.75);font-weight:400;margin:0;}
.hero-badge{display:inline-block;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);color:#FFF;font-size:11px;font-weight:600;padding:3px 10px;border-radius:999px;margin-right:6px;margin-top:10px;letter-spacing:.04em;}
.kpi{background:#FFF;border:1.5px solid #E2E8F0;border-radius:14px;padding:18px 22px;box-shadow:0 1px 6px rgba(0,0,0,.04);transition:box-shadow .2s,transform .2s;position:relative;overflow:hidden;}
.kpi:hover{box-shadow:0 8px 24px rgba(37,99,235,.10);transform:translateY(-2px);}
.kpi::after{content:'';position:absolute;top:0;left:0;width:4px;height:100%;border-radius:4px 0 0 4px;}
.kpi.blue::after{background:#2563EB;}.kpi.green::after{background:#10B981;}.kpi.amber::after{background:#F59E0B;}
.kpi.red::after{background:#EF4444;}.kpi.violet::after{background:#8B5CF6;}.kpi.cyan::after{background:#06B6D4;}
.kpi-label{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#64748B;margin-bottom:8px;}
.kpi-val{font-size:28px;font-weight:800;color:#0F172A;line-height:1;}
.kpi-sub{font-size:11px;color:#94A3B8;margin-top:4px;}
.sec-title{font-size:17px;font-weight:800;color:#0F172A;margin:28px 0 14px;display:flex;align-items:center;gap:8px;}
.sec-title::after{content:'';flex:1;height:1.5px;background:linear-gradient(90deg,#E2E8F0 0%,transparent 100%);border-radius:2px;}
.step-label{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;background:#2563EB;color:#FFF;font-size:12px;font-weight:800;margin-right:8px;flex-shrink:0;}
.info-box,.warn-box,.ok-box,.danger-box{border-radius:10px;padding:13px 17px;font-size:13.5px;margin-bottom:16px;font-weight:500;}
.info-box{background:#EFF6FF;border-left:4px solid #2563EB;color:#1E40AF;}
.warn-box{background:#FFFBEB;border-left:4px solid #F59E0B;color:#92400E;}
.ok-box{background:#F0FDF4;border-left:4px solid #22C55E;color:#166534;}
.danger-box{background:#FEF2F2;border-left:4px solid #EF4444;color:#991B1B;}
.pill{display:inline-block;padding:3px 11px;border-radius:999px;font-size:12.5px;font-weight:700;}
.pill-green{background:#D1FAE5;color:#065F46;}.pill-amber{background:#FEF3C7;color:#92400E;}
.pill-red{background:#FEE2E2;color:#991B1B;}.pill-blue{background:#DBEAFE;color:#1D4ED8;}
.stbl{width:100%;border-collapse:collapse;font-size:13.5px;background:#FFF;border-radius:12px;overflow:hidden;box-shadow:0 1px 6px rgba(0,0,0,.05);margin-bottom:20px;}
.stbl thead tr{background:#F8FAFC;}
.stbl th{padding:11px 15px;text-align:left;font-size:11px;font-weight:800;letter-spacing:.07em;text-transform:uppercase;color:#475569;border-bottom:1.5px solid #E2E8F0;}
.stbl td{padding:10px 15px;color:#334155;border-bottom:1px solid #F1F5F9;}
.stbl tr:last-child td{border-bottom:none;}.stbl tr:hover td{background:#F8FAFC;}
[data-baseweb="tab-list"]{gap:6px;background:#F1F5F9;padding:4px;border-radius:12px;}
[data-baseweb="tab"]{background:transparent!important;border:none!important;border-radius:8px!important;font-family:'Plus Jakarta Sans',sans-serif!important;font-size:13.5px!important;font-weight:600!important;color:#64748B!important;padding:8px 18px!important;transition:all .2s!important;}
[aria-selected="true"][data-baseweb="tab"]{background:#FFF!important;color:#2563EB!important;box-shadow:0 1px 6px rgba(0,0,0,.08)!important;}
[data-testid="stTabPanel"]{padding-top:20px;}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  PALETTE
# ════════════════════════════════════════════════════════════════════
C = dict(primary="#2563EB", success="#10B981", warning="#F59E0B",
         danger="#EF4444", violet="#8B5CF6", cyan="#06B6D4",
         slate="#64748B", bg="#FFFFFF", grid="#F1F5F9",
         text="#0F172A", yes_clr="#EF4444", no_clr="#10B981")

MODEL_CLR = {
    "Decision Tree"      :"#2563EB","Random Forest"      :"#10B981",
    "Gradient Boosting"  :"#F59E0B","Logistic Regression":"#8B5CF6",
    "SVM"                :"#06B6D4","KNN"                :"#F97316",
    "Naive Bayes"        :"#EC4899",
}

BL = dict(paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
          font=dict(family="Plus Jakarta Sans", color=C["text"], size=12),
          margin=dict(l=10,r=10,t=48,b=10),
          hoverlabel=dict(bgcolor="#FFF",font_size=12,
                          font_family="Plus Jakarta Sans",bordercolor="#E2E8F0"))

def L(**kw):
    d={**BL}; d.update(kw); return d

def _badge(v):
    p=v*100
    if p>=90: return f'<span class="pill pill-green">{p:.1f}%</span>'
    elif p>=75: return f'<span class="pill pill-amber">{p:.1f}%</span>'
    return f'<span class="pill pill-red">{p:.1f}%</span>'

# ════════════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def build_pipeline(csv_bytes):
    df_raw = pd.read_csv(io.BytesIO(csv_bytes))
    df_raw.columns = df_raw.columns.str.strip()
    basic = dict(shape=df_raw.shape,
                 dtypes=df_raw.dtypes.astype(str).to_dict(),
                 null_before=df_raw.isnull().sum().to_dict(),
                 describe=df_raw.describe(include="all"))
    df = df_raw.copy()
    for col in df.columns:
        if df[col].isnull().sum()==0: continue
        df[col].fillna(df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0], inplace=True)
    le_map, mapping = {}, []
    df_enc = df.copy()
    for col in df_enc.columns:
        if df_enc[col].dtype==object or str(df_enc[col].dtype)=="str":
            le=LabelEncoder(); df_enc[col]=le.fit_transform(df_enc[col].astype(str)); le_map[col]=le
            for o,e in zip(le.classes_,le.transform(le.classes_)):
                mapping.append({"Column":col,"Original":o,"Encoded":int(e)})
    LABEL="LUNG_CANCER"
    X=df_enc.drop(columns=[LABEL]); y=df_enc[LABEL]
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    corr=df_enc.corr()[LABEL].drop(LABEL).abs().sort_values(ascending=False)
    full_corr=df_enc.corr()
    chi2_rows=[]
    for col in X.columns:
        ct=pd.crosstab(df_enc[col],df_enc[LABEL])
        c2,p,_,_=chi2_contingency(ct)
        chi2_rows.append({"Feature":col,"Chi2":round(c2,3),"p-value":round(p,4),
                           "Significant":"Yes" if p<0.05 else "No"})
    return dict(df_raw=df_raw,df=df,df_enc=df_enc,basic=basic,
                map_df=pd.DataFrame(mapping),le_map=le_map,
                X=X,y=y,X_tr=X_tr,X_te=X_te,y_tr=y_tr,y_te=y_te,
                LABEL=LABEL,corr=corr,full_corr=full_corr,
                chi2_df=pd.DataFrame(chi2_rows).sort_values("Chi2",ascending=False),
                features=list(X.columns))

# ════════════════════════════════════════════════════════════════════
#  CLASSIFIERS
# ════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_classifiers(X_tr,X_te,y_tr,y_te,dt_d,rf_n,gb_n,_feats):
    defs={"Decision Tree":DecisionTreeClassifier(random_state=42,max_depth=dt_d),
          "Random Forest":RandomForestClassifier(n_estimators=rf_n,random_state=42),
          "Gradient Boosting":GradientBoostingClassifier(n_estimators=gb_n,random_state=42),
          "Logistic Regression":LogisticRegression(max_iter=1000,random_state=42),
          "SVM":SVC(probability=True,random_state=42),
          "KNN":KNeighborsClassifier(n_neighbors=5),
          "Naive Bayes":GaussianNB()}
    res={}
    for name,m in defs.items():
        m.fit(X_tr,y_tr); yp_tr=m.predict(X_tr); yp_te=m.predict(X_te)
        try: prob=m.predict_proba(X_te)[:,1]
        except: prob=None
        cv=cross_val_score(m,X_tr,y_tr,cv=StratifiedKFold(5),scoring="accuracy")
        res[name]=dict(model=m,train_acc=accuracy_score(y_tr,yp_tr),
                       test_acc=accuracy_score(y_te,yp_te),
                       precision=precision_score(y_te,yp_te,average="weighted",zero_division=0),
                       recall=recall_score(y_te,yp_te,average="weighted",zero_division=0),
                       f1=f1_score(y_te,yp_te,average="weighted",zero_division=0),
                       auc=roc_auc_score(y_te,prob) if prob is not None else None,
                       cm=confusion_matrix(y_te,yp_te),yp_te=yp_te,prob=prob,cv=cv,
                       imp=m.feature_importances_ if hasattr(m,"feature_importances_") else None)
    return res

# ════════════════════════════════════════════════════════════════════
#  ARM
# ════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_arm(df_enc,min_sup,min_conf,min_lift):
    sym=[c for c in df_enc.columns if c!="LUNG_CANCER"]
    df_b=df_enc[sym].copy()
    for col in sym: df_b[col]=(df_b[col]>=2).astype(bool)
    df_b["LUNG_CANCER_YES"]=(df_enc["LUNG_CANCER"]==1)
    freq=apriori(df_b,min_support=min_sup,use_colnames=True,max_len=4)
    if freq.empty: return freq,pd.DataFrame()
    rules=association_rules(freq,metric="confidence",min_threshold=min_conf,num_itemsets=len(freq))
    rules=rules[rules["lift"]>=min_lift].copy()
    rules.sort_values("lift",ascending=False,inplace=True); rules.reset_index(drop=True,inplace=True)
    rules["antecedents"]=rules["antecedents"].apply(lambda s:", ".join(sorted(s)))
    rules["consequents"]=rules["consequents"].apply(lambda s:", ".join(sorted(s)))
    return freq,rules

# ════════════════════════════════════════════════════════════════════
#  BIAS
# ════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_bias(df_raw,df_enc,_model,label,features):
    Xa=df_enc[features]; ya=df_enc[label]; yp=_model.predict(Xa)
    rows=[]
    def _add(gc,gv,mask):
        if mask.sum()<5: return
        tp=int(((ya[mask]==1)&(yp[mask]==1)).sum())
        fn=int(((ya[mask]==1)&(yp[mask]==0)).sum())
        fp=int(((ya[mask]==0)&(yp[mask]==1)).sum())
        tn=int(((ya[mask]==0)&(yp[mask]==0)).sum())
        tpr=tp/(tp+fn) if tp+fn>0 else 0
        fpr=fp/(fp+tn) if fp+tn>0 else 0
        rows.append({"Group Column":gc,"Group Value":str(gv),"N":int(mask.sum()),
                     "Accuracy":accuracy_score(ya[mask],yp[mask]),
                     "Positive Rate":yp[mask].mean(),"TPR":tpr,"FPR":fpr})
    if "GENDER" in df_raw.columns:
        for g in df_raw["GENDER"].dropna().unique(): _add("GENDER",g,(df_raw["GENDER"]==g).values)
    if "AGE" in df_raw.columns:
        df2=df_raw.copy(); df2["_ag"]=pd.cut(df2["AGE"],bins=[0,40,55,70,200],labels=["<40","40-55","56-70",">70"])
        for ag in ["<40","40-55","56-70",">70"]: _add("AGE GROUP",ag,(df2["_ag"]==ag).values)
    if "SMOKING" in df_raw.columns:
        for sv in df_raw["SMOKING"].dropna().unique():
            _add("SMOKING","Smoker" if sv==2 else "Non-Smoker",(df_raw["SMOKING"]==sv).values)
    return pd.DataFrame(rows)

# ════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ════════════════════════════════════════════════════════════════════
def cm_fig(cm,class_names,title,h=310):
    quad={(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}
    txt=[[f"{cm[i,j]}<br><sup>{quad.get((i,j),'')}</sup>" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    fig=go.Figure(go.Heatmap(z=cm,x=[f"Pred {c}" for c in class_names],y=[f"True {c}" for c in class_names],
        text=txt,texttemplate="%{text}",colorscale=[[0,"#EFF6FF"],[0.5,"#93C5FD"],[1,"#1D4ED8"]],
        showscale=False,hoverongaps=False,textfont=dict(size=17,color="#0F172A")))
    fig.update_layout(**L(title=dict(text=title,font=dict(size=13)),
        xaxis=dict(title="Predicted",tickfont=dict(size=11)),
        yaxis=dict(title="Actual",tickfont=dict(size=11),autorange="reversed"),
        height=h,margin=dict(l=60,r=10,t=50,b=50)))
    return fig

def feat_fig(imp,features,title,color=C["primary"],h=370):
    idx=np.argsort(imp)[-12:]; feats=np.array(features)[idx]; vals=imp[idx]
    fig=go.Figure(go.Bar(x=vals,y=feats,orientation="h",
        marker=dict(color=vals,colorscale=[[0,"#DBEAFE"],[1,color]],showscale=False,line=dict(width=0)),
        text=[f"{v:.3f}" for v in vals],textposition="outside",textfont=dict(size=11,color=C["slate"])))
    fig.update_layout(**L(title=dict(text=title,font=dict(size=13)),
        xaxis=dict(title="Importance",gridcolor=C["grid"],zeroline=False,range=[0,vals.max()*1.28]),
        yaxis=dict(tickfont=dict(size=11)),height=h,margin=dict(l=10,r=60,t=50,b=30)))
    return fig

# ════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:10px 0 18px;'><div style='font-size:36px;'>🫁</div><div style='font-size:15px;font-weight:800;color:#0F172A;'>Lung Cancer ML</div><div style='font-size:11px;color:#94A3B8;'>Detection Intelligence v2</div></div>",unsafe_allow_html=True)
    st.markdown("---")
    uploaded=st.file_uploader("Upload CSV",type="csv",label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**🌳 Tree Classifiers**")
    dt_depth=st.slider("Decision Tree depth",2,15,6)
    rf_trees=st.slider("Random Forest estimators",50,500,150,50)
    gb_trees=st.slider("Gradient Boost estimators",50,300,150,50)
    st.markdown("---")
    st.markdown("**🔗 Association Rules**")
    min_sup=st.slider("Min Support",0.10,0.80,0.30,0.05)
    min_conf=st.slider("Min Confidence",0.30,1.00,0.60,0.05)
    min_lift=st.slider("Min Lift",1.00,5.00,1.00,0.10)
    st.markdown("---")
    st.caption("Streamlit · scikit-learn · mlxtend · Plotly")

# ════════════════════════════════════════════════════════════════════
#  LOAD
# ════════════════════════════════════════════════════════════════════
with st.spinner("Loading data…"):
    try:
        csv_bytes=uploaded.read() if uploaded else open("survey_lung_cancer.csv","rb").read()
        P=build_pipeline(csv_bytes)
    except FileNotFoundError:
        st.error("survey_lung_cancer.csv not found. Upload via sidebar."); st.stop()

with st.spinner("Training 7 models…"):
    CLF=run_classifiers(P["X_tr"],P["X_te"],P["y_tr"],P["y_te"],dt_depth,rf_trees,gb_trees,P["features"])

LABEL=P["LABEL"]
CLASS_NAMES=P["le_map"][LABEL].classes_ if LABEL in P["le_map"] else ["NO","YES"]
best_name=max(CLF,key=lambda k:CLF[k]["test_acc"])
best_acc=CLF[best_name]["test_acc"]; best_f1=CLF[best_name]["f1"]
yes_n=int((P["df_raw"][LABEL]=="YES").sum()); no_n=int((P["df_raw"][LABEL]=="NO").sum())
pos_pct=yes_n/P["basic"]["shape"][0]

# ════════════════════════════════════════════════════════════════════
#  HERO + KPIs
# ════════════════════════════════════════════════════════════════════
st.markdown(f"""<div class="hero">
  <div class="hero-title">🫁 Lung Cancer Detection — ML Intelligence Dashboard v2</div>
  <div class="hero-sub">25+ interactive graph types · 7 models · Classification · Regression · ARM · Bias</div>
  <div style='margin-top:12px;'>
    <span class="hero-badge">309 Patients</span><span class="hero-badge">{len(P['features'])} Features</span>
    <span class="hero-badge">7 Models Benchmarked</span><span class="hero-badge">Best: {best_name} · {best_acc:.1%}</span>
  </div></div>""",unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6=st.columns(6)
for col,clr,lbl,val,sub in [(c1,"blue","Total Records",P["basic"]["shape"][0],"patients"),
    (c2,"red","Cancer +ve",f"{yes_n} ({pos_pct:.0%})","LUNG_CANCER=YES"),
    (c3,"green","Cancer −ve",no_n,"LUNG_CANCER=NO"),(c4,"violet","Features",len(P["features"]),"predictors"),
    (c5,"amber","Best Test Acc",f"{best_acc:.1%}",best_name),(c6,"cyan","Best F1",f"{best_f1:.1%}","weighted avg")]:
    col.markdown(f"""<div class="kpi {clr}"><div class="kpi-label">{lbl}</div>
      <div class="kpi-val">{val}</div><div class="kpi-sub">{sub}</div></div>""",unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════
T_OV,T_CLF,T_REG,T_ARM,T_BIAS=st.tabs([
    "📊  Data Overview","🌳  Classification","📈  Regression",
    "🔗  Association Rules","⚖️   Bias Detection"])


# ╔══════════════════════════════════════════════════════════════════
# ║  TAB 0 — DATA OVERVIEW
# ╚══════════════════════════════════════════════════════════════════
with T_OV:
    # ── Row 1: Donut · Funnel · Treemap ──────────────────────────────
    st.markdown('<div class="sec-title">Label Distribution & Symptom Analysis</div>',unsafe_allow_html=True)
    r1c1,r1c2,r1c3=st.columns(3)

    vc=P["df_raw"][LABEL].value_counts()
    with r1c1:
        fig=go.Figure(go.Pie(labels=vc.index,values=vc.values,hole=0.62,
            marker=dict(colors=[C["yes_clr"],C["no_clr"]],line=dict(color="#FFF",width=3)),
            textinfo="label+percent",textfont=dict(size=13)))
        fig.add_annotation(text=f"<b>{vc.sum()}</b><br>total",x=0.5,y=0.5,showarrow=False,
            font=dict(size=14,color=C["text"]))
        fig.update_layout(**L(title=dict(text="Class Split — Donut Chart",font=dict(size=13)),
            showlegend=True,legend=dict(orientation="h",yanchor="bottom",y=-0.18),height=290))
        st.plotly_chart(fig,use_container_width=True)

    sym_cols=[c for c in P["df_raw"].columns if c not in ["GENDER","AGE",LABEL]]
    with r1c2:
        prev=[]
        for sc in sym_cols:
            try:
                r=P["df_raw"][P["df_raw"][sc]==2][LABEL].value_counts(normalize=True).get("YES",0)
                prev.append((sc,round(r*100,1)))
            except: pass
        prev.sort(key=lambda x:x[1],reverse=True)
        top8=prev[:8]
        fig=go.Figure(go.Funnel(y=[x[0] for x in top8],x=[x[1] for x in top8],
            textinfo="value+percent initial",
            marker=dict(color=[C["primary"],C["violet"],C["cyan"],C["success"],
                                C["warning"],C["danger"],"#F97316","#EC4899"][:len(top8)])))
        fig.update_layout(**L(title=dict(text="Cancer Rate When Symptom Present (%)",font=dict(size=13)),height=290))
        st.plotly_chart(fig,use_container_width=True)

    with r1c3:
        sym_prev={sc:int((P["df_raw"][sc]==2).sum()) for sc in sym_cols if sc in P["df_raw"].columns}
        tm_df=pd.DataFrame(list(sym_prev.items()),columns=["Symptom","Count"])
        fig=px.treemap(tm_df,path=["Symptom"],values="Count",color="Count",
            color_continuous_scale=["#DBEAFE","#2563EB"])
        fig.update_layout(**L(title=dict(text="Symptom Prevalence — Treemap",font=dict(size=13)),
            height=290,margin=dict(l=5,r=5,t=48,b=5)))
        fig.update_traces(textfont=dict(size=12))
        st.plotly_chart(fig,use_container_width=True)

    # ── Full correlation heatmap ──────────────────────────────────────
    st.markdown('<div class="sec-title">Full Feature Correlation Matrix</div>',unsafe_allow_html=True)
    fc=P["full_corr"]
    fig=go.Figure(go.Heatmap(z=fc.values,x=fc.columns,y=fc.index,
        colorscale=[[0,"#DBEAFE"],[0.5,"#FFFFFF"],[1,"#1D4ED8"]],zmid=0,showscale=True,
        text=fc.round(2).values,texttemplate="%{text}",textfont=dict(size=9),hoverongaps=False))
    fig.update_layout(**L(title=dict(text="Pearson Correlation Matrix — All Features",font=dict(size=14)),
        xaxis=dict(tickangle=-45,tickfont=dict(size=9)),yaxis=dict(tickfont=dict(size=9)),
        height=520,margin=dict(l=100,r=10,t=50,b=100)))
    st.plotly_chart(fig,use_container_width=True)

    # ── Correlation bar + Chi2 bar ────────────────────────────────────
    co1,co2=st.columns(2)
    with co1:
        cs=P["corr"]
        fig=go.Figure(go.Bar(x=cs.values,y=cs.index,orientation="h",
            marker=dict(color=cs.values,colorscale=[[0,"#DBEAFE"],[1,C["primary"]]],
                        showscale=True,colorbar=dict(title="|r|",thickness=10)),
            text=[f"{v:.3f}" for v in cs.values],textposition="outside",textfont=dict(size=10)))
        fig.update_layout(**L(title=dict(text="|Correlation| with LUNG_CANCER",font=dict(size=13)),
            xaxis=dict(range=[0,cs.max()*1.25],gridcolor=C["grid"]),height=400))
        st.plotly_chart(fig,use_container_width=True)
    with co2:
        chi=P["chi2_df"].sort_values("Chi2",ascending=True)
        colors=[C["success"] if s=="Yes" else C["danger"] for s in chi["Significant"]]
        fig=go.Figure(go.Bar(x=chi["Chi2"],y=chi["Feature"],orientation="h",
            marker_color=colors,
            text=[f"χ²={v:.1f}  p={p:.3f}" for v,p in zip(chi["Chi2"],chi["p-value"])],
            textposition="outside",textfont=dict(size=10)))
        fig.update_layout(**L(title=dict(text="Chi-Square (green=significant p<0.05)",font=dict(size=13)),
            xaxis=dict(gridcolor=C["grid"]),height=400))
        st.plotly_chart(fig,use_container_width=True)

    # ── Violin + Grouped symptom mean bars ───────────────────────────
    st.markdown('<div class="sec-title">Feature Distributions by Cancer Status</div>',unsafe_allow_html=True)
    vio1,vio2=st.columns(2)
    with vio1:
        fig=go.Figure()
        for val,clr,nm in [("YES",C["yes_clr"],"Cancer YES"),("NO",C["no_clr"],"Cancer NO")]:
            ages=P["df_raw"][P["df_raw"][LABEL]==val]["AGE"]
            fig.add_trace(go.Violin(y=ages,name=nm,box_visible=True,meanline_visible=True,
                fillcolor=clr,opacity=0.7,line_color=clr))
        fig.update_layout(**L(title=dict(text="AGE — Violin + Box Plot",font=dict(size=13)),
            yaxis=dict(title="Age",gridcolor=C["grid"]),violingap=0.3,violinmode="overlay",height=340))
        st.plotly_chart(fig,use_container_width=True)
    with vio2:
        sym_means={}
        for sc in sym_cols:
            try:
                yes_m=P["df_raw"][P["df_raw"][LABEL]=="YES"][sc].mean()
                no_m =P["df_raw"][P["df_raw"][LABEL]=="NO"][sc].mean()
                sym_means[sc]={"YES":yes_m,"NO":no_m}
            except: pass
        sm_df=pd.DataFrame(sym_means).T.reset_index(); sm_df.columns=["Symptom","YES Mean","NO Mean"]
        fig=go.Figure()
        fig.add_trace(go.Bar(x=sm_df["Symptom"],y=sm_df["YES Mean"],name="Cancer YES",marker_color=C["yes_clr"],opacity=0.85))
        fig.add_trace(go.Bar(x=sm_df["Symptom"],y=sm_df["NO Mean"],name="Cancer NO",marker_color=C["no_clr"],opacity=0.85))
        fig.update_layout(**L(barmode="group",title=dict(text="Mean Symptom Score YES vs NO",font=dict(size=13)),
            xaxis=dict(tickangle=-40,tickfont=dict(size=9)),
            yaxis=dict(title="Mean",gridcolor=C["grid"]),
            legend=dict(orientation="h",yanchor="bottom",y=1.02),height=340))
        st.plotly_chart(fig,use_container_width=True)

    # ── Parallel coordinates ─────────────────────────────────────────
    st.markdown('<div class="sec-title">Parallel Coordinates — Feature Patterns</div>',unsafe_allow_html=True)
    st.markdown('<div class="info-box">Each line = one patient. Colour = cancer status. Drag axes to reorder and find separation.</div>',unsafe_allow_html=True)
    pc_cols=["AGE","SMOKING","YELLOW_FINGERS","ANXIETY","COUGHING","WHEEZING","ALLERGY","CHEST PAIN"]
    pc_ok=[c for c in pc_cols if c in P["df_enc"].columns]
    pc_df=P["df_enc"][pc_ok+[LABEL]].copy()
    fig=px.parallel_coordinates(pc_df,color=LABEL,dimensions=pc_ok,
        color_continuous_scale=[[0,C["no_clr"]],[1,C["yes_clr"]]])
    fig.update_layout(**L(title=dict(text="Parallel Coordinates — Patient Feature Profiles",font=dict(size=14)),height=390))
    st.plotly_chart(fig,use_container_width=True)

    # ── PCA 2D scatter ───────────────────────────────────────────────
    st.markdown('<div class="sec-title">PCA 2D — Patient Clusters</div>',unsafe_allow_html=True)
    pca=PCA(n_components=2,random_state=42); comps=pca.fit_transform(P["X"])
    pca_df=pd.DataFrame(comps,columns=["PC1","PC2"]); pca_df[LABEL]=P["y"].values
    # decode label if needed
    if P["y"].max()<=1:
        lmap={0:"NO",1:"YES"}; pca_df["Status"]=pca_df[LABEL].map(lmap)
    else:
        pca_df["Status"]=pca_df[LABEL].astype(str)
    fig=px.scatter(pca_df,x="PC1",y="PC2",color="Status",
        color_discrete_map={"YES":C["yes_clr"],"NO":C["no_clr"]},opacity=0.72,
        title=f"PCA 2D — PC1 {pca.explained_variance_ratio_[0]:.1%} | PC2 {pca.explained_variance_ratio_[1]:.1%}")
    fig.update_traces(marker=dict(size=7,line=dict(width=1,color="#FFF")))
    fig.update_layout(**L(xaxis=dict(gridcolor=C["grid"]),yaxis=dict(gridcolor=C["grid"]),height=400))
    st.plotly_chart(fig,use_container_width=True)

    # ── Sunburst: Gender × Cancer × Smoking ──────────────────────────
    st.markdown('<div class="sec-title">Sunburst — Gender × Cancer × Smoking</div>',unsafe_allow_html=True)
    if all(c in P["df_raw"].columns for c in ["GENDER","SMOKING"]):
        sb_df=P["df_raw"].copy(); sb_df["Smoking"]=sb_df["SMOKING"].map({1:"Non-Smoker",2:"Smoker"})
        sb_grp=sb_df.groupby(["GENDER",LABEL,"Smoking"]).size().reset_index(name="Count")
        fig=px.sunburst(sb_grp,path=["GENDER",LABEL,"Smoking"],values="Count",
            color="Count",color_continuous_scale=["#DBEAFE","#2563EB"],
            title="Sunburst: Gender → Cancer Status → Smoking")
        fig.update_layout(**L(height=420,margin=dict(l=5,r=5,t=48,b=5)))
        st.plotly_chart(fig,use_container_width=True)

    # ── Encoding map ─────────────────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="step-label">4</span>Label Encoding Map</div>',unsafe_allow_html=True)
    ec1,ec2=st.columns([1,2])
    with ec1:
        st.dataframe(P["map_df"],use_container_width=True,height=180,hide_index=True)
        st.download_button("⬇️ Download Encoded CSV",P["df_enc"].to_csv(index=False).encode(),
            "lung_cancer_encoded.csv","text/csv",use_container_width=True)
    with ec2:
        st.dataframe(P["df_enc"].head(20),use_container_width=True,height=180)


# ╔══════════════════════════════════════════════════════════════════
# ║  TAB 1 — CLASSIFICATION
# ╚══════════════════════════════════════════════════════════════════
with T_CLF:
    st.markdown(f"""<div class="info-box"><b>Steps 5–6 complete.</b>
    Train: <b>{len(P['X_tr'])}</b> | Test: <b>{len(P['X_te'])}</b> | 80:20 stratified split.</div>""",
    unsafe_allow_html=True)

    # ── Metrics table ────────────────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="step-label">8</span>Performance — All 7 Models</div>',unsafe_allow_html=True)
    rows=[{"Model":nm,"train_acc":r["train_acc"],"test_acc":r["test_acc"],
           "precision":r["precision"],"recall":r["recall"],"f1":r["f1"],
           "auc":r["auc"],"cv":r["cv"].mean()} for nm,r in CLF.items()]
    acc_df=pd.DataFrame(rows)
    tbl="""<table class="stbl"><thead><tr><th>Model</th><th>Train</th><th>Test</th>
        <th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th>5-Fold CV</th></tr></thead><tbody>"""
    for _,r in acc_df.iterrows():
        tbl+=f"""<tr><td><b style='color:{MODEL_CLR.get(r["Model"],C["text"])};'>{r["Model"]}</b></td>
          <td>{_badge(r["train_acc"])}</td><td>{_badge(r["test_acc"])}</td>
          <td>{r["precision"]*100:.1f}%</td><td>{r["recall"]*100:.1f}%</td><td>{r["f1"]*100:.1f}%</td>
          <td>{"—" if r["auc"] is None else f'{r["auc"]:.3f}'}</td><td>{r["cv"]*100:.1f}%</td></tr>"""
    tbl+="</tbody></table>"; st.markdown(tbl,unsafe_allow_html=True)

    # ── Grouped bar + Waterfall delta ────────────────────────────────
    cl1,cl2=st.columns(2)
    with cl1:
        fig=go.Figure()
        for met,lbl,clr in [("test_acc","Test Acc",C["primary"]),("precision","Precision",C["success"]),
                              ("recall","Recall",C["warning"]),("f1","F1",C["violet"])]:
            fig.add_trace(go.Bar(name=lbl,x=acc_df["Model"],y=acc_df[met]*100,marker_color=clr,
                text=[f"{v:.0f}%" for v in acc_df[met]*100],textposition="outside",textfont=dict(size=9)))
        fig.update_layout(**L(barmode="group",title=dict(text="Metric Comparison",font=dict(size=14)),
            yaxis=dict(range=[0,115],gridcolor=C["grid"],title="%"),xaxis=dict(tickangle=-12,tickfont=dict(size=10)),
            legend=dict(orientation="h",yanchor="bottom",y=1.02),height=390))
        st.plotly_chart(fig,use_container_width=True)
    with cl2:
        mean_acc=acc_df["test_acc"].mean(); deltas=acc_df["test_acc"]-mean_acc
        fig=go.Figure(go.Bar(x=acc_df["Model"],y=deltas*100,
            marker_color=[C["success"] if d>=0 else C["danger"] for d in deltas],
            text=[f"{d*100:+.1f}%" for d in deltas],textposition="outside",textfont=dict(size=10)))
        fig.add_hline(y=0,line_color=C["slate"],line_width=1.5)
        fig.update_layout(**L(title=dict(text=f"Test Acc Delta from Mean ({mean_acc:.1%})",font=dict(size=14)),
            yaxis=dict(title="Δ (%)",gridcolor=C["grid"]),xaxis=dict(tickangle=-12,tickfont=dict(size=10)),height=390))
        st.plotly_chart(fig,use_container_width=True)

    # ── ROC + PR curves ──────────────────────────────────────────────
    r1c,r2c=st.columns(2)
    with r1c:
        fig=go.Figure()
        fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line=dict(dash="dot",color=C["slate"],width=1))
        for nm,r in CLF.items():
            if r["prob"] is not None:
                fpr,tpr,_=roc_curve(P["y_te"],r["prob"])
                fig.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{nm} ({r['auc']:.3f})",
                    line=dict(color=MODEL_CLR[nm],width=2.5),mode="lines"))
        fig.update_layout(**L(title=dict(text="ROC Curves",font=dict(size=14)),
            xaxis=dict(title="FPR",gridcolor=C["grid"],range=[0,1]),
            yaxis=dict(title="TPR",gridcolor=C["grid"],range=[0,1.02]),
            legend=dict(x=0.55,y=0.05,bgcolor="#FFF",bordercolor="#E2E8F0",borderwidth=1,font=dict(size=10)),height=400))
        st.plotly_chart(fig,use_container_width=True)
    with r2c:
        fig=go.Figure()
        for nm,r in CLF.items():
            if r["prob"] is not None:
                prec,rec,_=precision_recall_curve(P["y_te"],r["prob"])
                ap=average_precision_score(P["y_te"],r["prob"])
                fig.add_trace(go.Scatter(x=rec,y=prec,name=f"{nm} (AP={ap:.3f})",
                    line=dict(color=MODEL_CLR[nm],width=2.5),mode="lines"))
        fig.update_layout(**L(title=dict(text="Precision-Recall Curves",font=dict(size=14)),
            xaxis=dict(title="Recall",gridcolor=C["grid"]),yaxis=dict(title="Precision",gridcolor=C["grid"]),
            legend=dict(x=0.01,y=0.01,bgcolor="#FFF",bordercolor="#E2E8F0",borderwidth=1,font=dict(size=10)),height=400))
        st.plotly_chart(fig,use_container_width=True)

    # ── Confusion matrices ───────────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="step-label">9</span>Confusion Matrices</div>',unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>TP</b>=True Positive · <b>TN</b>=True Negative · <b>FP</b>=False Alarm · <b>FN</b>=Missed Cancer</div>',unsafe_allow_html=True)
    nm_list=list(CLF.keys())
    for start in range(0,7,4):
        chunk=nm_list[start:start+4]; cols=st.columns(len(chunk))
        for col_w,nm in zip(cols,chunk):
            with col_w: st.plotly_chart(cm_fig(CLF[nm]["cm"],CLASS_NAMES,nm),use_container_width=True)

    # ── Feature importance ───────────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="step-label">10</span>Feature Importances</div>',unsafe_allow_html=True)
    tree_ms={k:v for k,v in CLF.items() if v["imp"] is not None}
    fi_cols=st.columns(len(tree_ms))
    for col_w,(nm,r) in zip(fi_cols,tree_ms.items()):
        with col_w: st.plotly_chart(feat_fig(r["imp"],P["features"],nm,color=MODEL_CLR[nm]),use_container_width=True)

    # ── Feature importance heatmap ───────────────────────────────────
    st.markdown('<div class="sec-title">Feature Importance Heatmap — Tree Models</div>',unsafe_allow_html=True)
    imp_mat=np.array([CLF[nm]["imp"] for nm in tree_ms.keys()])
    fig=go.Figure(go.Heatmap(z=imp_mat,x=P["features"],y=list(tree_ms.keys()),
        colorscale=[[0,"#EFF6FF"],[0.5,"#93C5FD"],[1,"#1D4ED8"]],showscale=True,
        text=np.round(imp_mat,3),texttemplate="%{text}",textfont=dict(size=9)))
    fig.update_layout(**L(title=dict(text="Importance Heatmap — Models × Features",font=dict(size=14)),
        xaxis=dict(tickangle=-40,tickfont=dict(size=9)),yaxis=dict(tickfont=dict(size=11)),
        height=260,margin=dict(l=160,r=30,t=50,b=100)))
    st.plotly_chart(fig,use_container_width=True)

    # ── Learning curves ──────────────────────────────────────────────
    st.markdown('<div class="sec-title">Learning Curves — Bias vs Variance</div>',unsafe_allow_html=True)
    lc_sel=st.selectbox("Model for learning curve",["Decision Tree","Random Forest","Gradient Boosting"],key="lc")
    @st.cache_data(show_spinner=False)
    def _lc(nm,Xtr,ytr,dt_d,rf_n,gb_n):
        m={"Decision Tree":DecisionTreeClassifier(random_state=42,max_depth=dt_d),
           "Random Forest":RandomForestClassifier(n_estimators=rf_n,random_state=42),
           "Gradient Boosting":GradientBoostingClassifier(n_estimators=gb_n,random_state=42)}[nm]
        ts,tr,va=learning_curve(m,Xtr,ytr,cv=StratifiedKFold(5),
            train_sizes=np.linspace(0.2,1.0,8),scoring="accuracy",n_jobs=-1)[:3]
        return ts,tr.mean(1),va.mean(1),tr.std(1),va.std(1)
    ts,tr_m,va_m,tr_s,va_s=_lc(lc_sel,P["X_tr"],P["y_tr"],dt_depth,rf_trees,gb_trees)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ts,y=tr_m*100,name="Train",mode="lines+markers",
        line=dict(color=C["primary"],width=2.5),
        error_y=dict(type="data",array=tr_s*100,visible=True,color=C["primary"],thickness=1)))
    fig.add_trace(go.Scatter(x=ts,y=va_m*100,name="Validation",mode="lines+markers",
        line=dict(color=C["success"],width=2.5),
        error_y=dict(type="data",array=va_s*100,visible=True,color=C["success"],thickness=1)))
    fig.update_layout(**L(title=dict(text=f"Learning Curve — {lc_sel}",font=dict(size=14)),
        xaxis=dict(title="Training Samples",gridcolor=C["grid"]),
        yaxis=dict(title="Accuracy (%)",gridcolor=C["grid"]),
        legend=dict(orientation="h",yanchor="bottom",y=1.02),height=360))
    st.plotly_chart(fig,use_container_width=True)

    # ── Calibration curves ───────────────────────────────────────────
    st.markdown('<div class="sec-title">Calibration (Reliability) Diagrams</div>',unsafe_allow_html=True)
    st.markdown('<div class="info-box">Points on diagonal = perfectly calibrated. Above = under-confident; below = over-confident.</div>',unsafe_allow_html=True)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dot",color=C["slate"]),name="Perfect"))
    for nm,r in CLF.items():
        if r["prob"] is not None:
            try:
                fc_,mc_=calibration_curve(P["y_te"],r["prob"],n_bins=6)
                fig.add_trace(go.Scatter(x=mc_,y=fc_,mode="lines+markers",name=nm,
                    line=dict(color=MODEL_CLR[nm],width=2),marker=dict(size=7)))
            except: pass
    fig.update_layout(**L(title=dict(text="Calibration Curves — All Models",font=dict(size=14)),
        xaxis=dict(title="Mean Predicted Probability",gridcolor=C["grid"],range=[0,1]),
        yaxis=dict(title="Fraction of Positives",gridcolor=C["grid"],range=[0,1]),height=380))
    st.plotly_chart(fig,use_container_width=True)

    # ── CV box plots ─────────────────────────────────────────────────
    st.markdown('<div class="sec-title">5-Fold CV Distribution — Box Plots</div>',unsafe_allow_html=True)
    fig=go.Figure()
    for nm,r in CLF.items():
        fig.add_trace(go.Box(y=r["cv"]*100,name=nm,marker_color=MODEL_CLR[nm],boxmean="sd",line=dict(width=2)))
    fig.update_layout(**L(title=dict(text="Cross-Validation Accuracy Distribution",font=dict(size=14)),
        yaxis=dict(title="Accuracy (%)",gridcolor=C["grid"]),height=360))
    st.plotly_chart(fig,use_container_width=True)

    # ── Threshold sensitivity ────────────────────────────────────────
    st.markdown('<div class="sec-title">Decision Threshold Sensitivity</div>',unsafe_allow_html=True)
    if CLF[best_name]["prob"] is not None:
        thresholds=np.linspace(0.05,0.95,50)
        th_p,th_r,th_f,th_a=[],[],[],[]
        for t in thresholds:
            yp_t=(CLF[best_name]["prob"]>=t).astype(int)
            th_p.append(precision_score(P["y_te"],yp_t,zero_division=0))
            th_r.append(recall_score(P["y_te"],yp_t,zero_division=0))
            th_f.append(f1_score(P["y_te"],yp_t,zero_division=0))
            th_a.append(accuracy_score(P["y_te"],yp_t))
        fig=go.Figure()
        for y_vals,lbl,clr in [(th_a,"Accuracy",C["primary"]),(th_p,"Precision",C["warning"]),
                                 (th_r,"Recall",C["yes_clr"]),(th_f,"F1",C["violet"])]:
            fig.add_trace(go.Scatter(x=thresholds*100,y=np.array(y_vals)*100,name=lbl,
                mode="lines",line=dict(color=clr,width=2.5)))
        fig.add_vline(x=50,line_dash="dash",line_color=C["slate"],
            annotation_text="Default 0.5",annotation_font=dict(size=11))
        fig.update_layout(**L(title=dict(text=f"Threshold Analysis — {best_name}",font=dict(size=14)),
            xaxis=dict(title="Threshold (%)",gridcolor=C["grid"]),
            yaxis=dict(title="Score (%)",gridcolor=C["grid"]),
            legend=dict(orientation="h",yanchor="bottom",y=1.02),height=380))
        st.plotly_chart(fig,use_container_width=True)

    # ── Per-class report ─────────────────────────────────────────────
    st.markdown('<div class="sec-title">Detailed Classification Report</div>',unsafe_allow_html=True)
    sel_m=st.selectbox("Pick model",list(CLF.keys()),key="clf_rpt")
    cr=classification_report(P["y_te"],CLF[sel_m]["yp_te"],target_names=CLASS_NAMES,output_dict=True)
    st.dataframe(pd.DataFrame(cr).T.round(3),use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════
# ║  TAB 2 — REGRESSION
# ╚══════════════════════════════════════════════════════════════════
with T_REG:
    st.markdown('<div class="info-box"><b>Regression-family classifiers</b> — Logistic Regression, SVM, KNN, Naive Bayes use linear/kernel/probabilistic decision boundaries.</div>',unsafe_allow_html=True)
    reg_names=["Logistic Regression","SVM","KNN","Naive Bayes"]

    rtbl="""<table class="stbl"><thead><tr><th>Model</th><th>Train</th><th>Test</th>
        <th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr></thead><tbody>"""
    for nm in reg_names:
        r=CLF[nm]
        rtbl+=f"""<tr><td><b style='color:{MODEL_CLR[nm]};'>{nm}</b></td>
          <td>{_badge(r["train_acc"])}</td><td>{_badge(r["test_acc"])}</td>
          <td>{r["precision"]*100:.1f}%</td><td>{r["recall"]*100:.1f}%</td>
          <td>{r["f1"]*100:.1f}%</td><td>{"—" if r["auc"] is None else f'{r["auc"]:.3f}'}</td></tr>"""
    rtbl+="</tbody></table>"; st.markdown(rtbl,unsafe_allow_html=True)

    # ── Radar + Bubble ───────────────────────────────────────────────
    rg1,rg2=st.columns(2)
    with rg1:
        cats=["Test Acc","Precision","Recall","F1","CV Mean"]
        fig=go.Figure()
        for nm in CLF.keys():
            r=CLF[nm]; vals=[r["test_acc"],r["precision"],r["recall"],r["f1"],r["cv"].mean()]; vals+=[vals[0]]
            fig.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],fill="toself",name=nm,
                line=dict(color=MODEL_CLR[nm],width=2),fillcolor=MODEL_CLR[nm],opacity=0.12))
        fig.update_layout(**L(polar=dict(radialaxis=dict(visible=True,range=[0.5,1.05],
            tickformat=".0%",gridcolor=C["grid"])),
            title=dict(text="Radar — All Models",font=dict(size=14)),
            legend=dict(orientation="h",yanchor="bottom",y=-0.28,font=dict(size=10)),height=470))
        st.plotly_chart(fig,use_container_width=True)
    with rg2:
        bub_df=pd.DataFrame([{"Model":nm,"Precision":CLF[nm]["precision"]*100,
            "Recall":CLF[nm]["recall"]*100,"F1":CLF[nm]["f1"]*100} for nm in CLF.keys()])
        fig=px.scatter(bub_df,x="Precision",y="Recall",size="F1",color="Model",text="Model",
            color_discrete_map=MODEL_CLR,size_max=50,title="Precision vs Recall (bubble = F1)")
        fig.update_traces(textposition="top center",marker=dict(line=dict(width=1.5,color="#FFF")))
        fig.update_layout(**L(xaxis=dict(title="Precision (%)",gridcolor=C["grid"]),
            yaxis=dict(title="Recall (%)",gridcolor=C["grid"]),showlegend=False,height=470))
        st.plotly_chart(fig,use_container_width=True)

    # ── Confusion matrices ───────────────────────────────────────────
    st.markdown('<div class="sec-title">Confusion Matrices — Regression Models</div>',unsafe_allow_html=True)
    rc=st.columns(4)
    for col_w,nm in zip(rc,reg_names):
        with col_w: st.plotly_chart(cm_fig(CLF[nm]["cm"],CLASS_NAMES,nm,h=270),use_container_width=True)

    # ── Horizontal ranking ───────────────────────────────────────────
    st.markdown('<div class="sec-title">All-Model Test Accuracy Ranking</div>',unsafe_allow_html=True)
    rank_df=pd.DataFrame([{"Model":nm,"Test Acc":CLF[nm]["test_acc"]*100} for nm in CLF]).sort_values("Test Acc",ascending=True)
    fig=go.Figure(go.Bar(x=rank_df["Test Acc"],y=rank_df["Model"],orientation="h",
        marker=dict(color=[MODEL_CLR[nm] for nm in rank_df["Model"]],line=dict(width=0)),
        text=[f"{v:.1f}%" for v in rank_df["Test Acc"]],textposition="outside",textfont=dict(size=12)))
    fig.update_layout(**L(title=dict(text="Test Accuracy Ranking",font=dict(size=14)),
        xaxis=dict(range=[0,115],gridcolor=C["grid"],title="%"),height=360))
    st.plotly_chart(fig,use_container_width=True)

    # ── LR coefficients + Odds ratios ────────────────────────────────
    st.markdown('<div class="sec-title">Logistic Regression — Coefficients & Odds Ratios</div>',unsafe_allow_html=True)
    lr=CLF["Logistic Regression"]["model"]; coef=lr.coef_[0]
    coef_df=pd.DataFrame({"Feature":P["features"],"Coef":coef}).sort_values("Coef",key=abs,ascending=False)
    rg3,rg4=st.columns(2)
    with rg3:
        fig=go.Figure(go.Bar(x=coef_df["Coef"],y=coef_df["Feature"],orientation="h",
            marker=dict(color=coef_df["Coef"],colorscale=[[0,C["yes_clr"]],[0.5,"#F8FAFC"],[1,C["primary"]]],
                        cmid=0,showscale=True)))
        fig.update_layout(**L(title=dict(text="LR Coefficients (signed)",font=dict(size=13)),
            xaxis=dict(title="Coefficient",gridcolor=C["grid"],zeroline=True,zerolinecolor=C["slate"]),height=380))
        st.plotly_chart(fig,use_container_width=True)
    with rg4:
        odds=np.exp(coef); od_df=pd.DataFrame({"Feature":P["features"],"OR":odds}).sort_values("OR",ascending=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=od_df["OR"],y=od_df["Feature"],mode="markers+text",
            text=[f"{v:.2f}" for v in od_df["OR"]],textposition="middle right",textfont=dict(size=10),
            marker=dict(size=13,color=od_df["OR"],colorscale=[[0,C["no_clr"]],[1,C["yes_clr"]]],
                        showscale=True,colorbar=dict(title="OR",thickness=10),line=dict(width=1,color="#FFF"))))
        fig.add_vline(x=1,line_dash="dash",line_color=C["slate"])
        fig.update_layout(**L(title=dict(text="Odds Ratios (OR>1 = ↑ cancer risk)",font=dict(size=13)),
            xaxis=dict(title="Odds Ratio",gridcolor=C["grid"]),height=380))
        st.plotly_chart(fig,use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════
# ║  TAB 3 — ASSOCIATION RULE MINING
# ╚══════════════════════════════════════════════════════════════════
with T_ARM:
    st.markdown('<div class="info-box"><b>Apriori</b> finds symptom co-occurrence patterns. High-Lift rules reveal combinations strongly linked to cancer diagnosis.</div>',unsafe_allow_html=True)
    with st.spinner("Running Apriori…"):
        freq_items,rules=run_arm(P["df_enc"],min_sup,min_conf,min_lift)

    if rules.empty:
        st.markdown('<div class="warn-box">No rules at current thresholds. Lower Min Support / Confidence in the sidebar.</div>',unsafe_allow_html=True)
    else:
        ka,kb,kc,kd,ke=st.columns(5)
        for col_w,clr,lbl,val,sub in [
            (ka,"blue","Rules",len(rules),"passing filters"),
            (kb,"green","Avg Conf",f"{rules['confidence'].mean():.1%}","mean confidence"),
            (kc,"amber","Max Lift",f"{rules['lift'].max():.2f}","strongest"),
            (kd,"violet","Avg Support",f"{rules['support'].mean():.1%}","itemset freq"),
            (ke,"cyan","Freq Itemsets",len(freq_items),"above min support")]:
            col_w.markdown(f"""<div class="kpi {clr}"><div class="kpi-label">{lbl}</div>
              <div class="kpi-val">{val}</div><div class="kpi-sub">{sub}</div></div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)

        # Rules table
        st.markdown('<div class="sec-title">All Association Rules</div>',unsafe_allow_html=True)
        disp=rules[["antecedents","consequents","support","confidence","lift","leverage","conviction"]].copy()
        disp["support"]=(disp["support"]*100).round(2).astype(str)+"%"
        disp["confidence"]=(disp["confidence"]*100).round(2).astype(str)+"%"
        for col in ["lift","leverage","conviction"]: disp[col]=disp[col].round(3)
        st.dataframe(disp,use_container_width=True,height=320)

        # ── Top-20 bar + 3D scatter ──────────────────────────────────
        arm1,arm2=st.columns(2)
        with arm1:
            top20=rules.head(20).copy(); top20["Rule"]=top20["antecedents"]+" → "+top20["consequents"]
            fig=go.Figure(go.Bar(x=top20["lift"],y=top20["Rule"],orientation="h",
                marker=dict(color=top20["lift"],colorscale=[[0,"#DBEAFE"],[1,C["primary"]]],
                            showscale=True,colorbar=dict(title="Lift",thickness=10)),
                text=[f"{v:.2f}" for v in top20["lift"]],textposition="outside",textfont=dict(size=9)))
            fig.update_layout(**L(title=dict(text="Top 20 Rules by Lift",font=dict(size=13)),
                xaxis=dict(title="Lift",gridcolor=C["grid"]),
                height=560,margin=dict(l=10,r=60,t=50,b=30)))
            st.plotly_chart(fig,use_container_width=True)
        with arm2:
            fig=px.scatter_3d(rules.head(60),x="support",y="confidence",z="lift",
                color="lift",size="lift",color_continuous_scale="Blues",
                hover_data=["antecedents","consequents"],title="3D: Support / Confidence / Lift")
            fig.update_layout(**L(height=560,scene=dict(
                xaxis=dict(title="Support",backgroundcolor="#FAFBFF"),
                yaxis=dict(title="Confidence",backgroundcolor="#FAFBFF"),
                zaxis=dict(title="Lift",backgroundcolor="#FAFBFF"))))
            st.plotly_chart(fig,use_container_width=True)

        # ── Support vs Confidence scatter ────────────────────────────
        st.markdown('<div class="sec-title">Support vs Confidence (bubble = Lift)</div>',unsafe_allow_html=True)
        fig=go.Figure(go.Scatter(x=rules["support"],y=rules["confidence"],mode="markers",
            marker=dict(size=rules["lift"]*5,color=rules["lift"],colorscale="Blues",showscale=True,
                        colorbar=dict(title="Lift",thickness=10),opacity=0.8,line=dict(width=1,color="#FFF")),
            text=[f"{a} → {c}<br>Lift {l:.2f}" for a,c,l in zip(rules["antecedents"],rules["consequents"],rules["lift"])],
            hovertemplate="%{text}<extra></extra>"))
        fig.update_layout(**L(xaxis=dict(title="Support",gridcolor=C["grid"]),
            yaxis=dict(title="Confidence",gridcolor=C["grid"]),height=420))
        st.plotly_chart(fig,use_container_width=True)

        # ── ARM Heatmap ──────────────────────────────────────────────
        st.markdown('<div class="sec-title">Association Lift Heatmap — Antecedent × Consequent</div>',unsafe_allow_html=True)
        top_r=rules.head(30).copy()
        ant_u=top_r["antecedents"].unique()[:12]; con_u=top_r["consequents"].unique()[:8]
        mat=pd.DataFrame(0.0,index=ant_u,columns=con_u)
        for _,row in top_r.iterrows():
            if row["antecedents"] in mat.index and row["consequents"] in mat.columns:
                mat.loc[row["antecedents"],row["consequents"]]=row["lift"]
        fig=go.Figure(go.Heatmap(z=mat.values,x=mat.columns,y=mat.index,
            colorscale=[[0,"#FFFFFF"],[0.5,"#93C5FD"],[1,"#1D4ED8"]],showscale=True,
            text=np.round(mat.values,2),texttemplate="%{text}",textfont=dict(size=9)))
        fig.update_layout(**L(title=dict(text="Lift Heatmap — Antecedents vs Consequents",font=dict(size=14)),
            xaxis=dict(tickangle=-35,tickfont=dict(size=9)),yaxis=dict(tickfont=dict(size=9)),
            height=460,margin=dict(l=250,r=20,t=50,b=120)))
        st.plotly_chart(fig,use_container_width=True)

        # ── LUNG_CANCER rules ────────────────────────────────────────
        lc_rules=rules[rules["consequents"].str.contains("LUNG_CANCER",na=False)]
        if not lc_rules.empty:
            st.markdown('<div class="sec-title">🫁 Rules → LUNG_CANCER_YES</div>',unsafe_allow_html=True)
            st.markdown('<div class="danger-box">High-lift rules predicting cancer-positive outcomes — highest clinical priority.</div>',unsafe_allow_html=True)
            dsp2=lc_rules[["antecedents","consequents","support","confidence","lift"]].copy()
            dsp2["support"]=(dsp2["support"]*100).round(2).astype(str)+"%"
            dsp2["confidence"]=(dsp2["confidence"]*100).round(2).astype(str)+"%"
            dsp2["lift"]=dsp2["lift"].round(3)
            st.dataframe(dsp2,use_container_width=True)

        with st.expander("📖 ARM metric guide"):
            st.markdown("""
| Metric | Formula | Meaning |
|---|---|---|
| **Support** | P(A∪B) | Frequency of itemset in data |
| **Confidence** | P(B\|A) | Probability of B given A |
| **Lift** | Conf/P(B) | >1 = positive link; 1 = independent |
| **Leverage** | P(A∪B)−P(A)·P(B) | Co-occurrence excess over chance |
| **Conviction** | (1−P(B))/(1−Conf) | Strength of implication; ∞=perfect |""")


# ╔══════════════════════════════════════════════════════════════════
# ║  TAB 4 — BIAS DETECTION
# ╚══════════════════════════════════════════════════════════════════
with T_BIAS:
    st.markdown('<div class="danger-box"><b>⚖️ Medical AI Bias:</b> High overall accuracy can mask systematic under-diagnosis in specific demographic groups. Every number below is backed by data.</div>',unsafe_allow_html=True)
    best_r=CLF[best_name]; bias_df=run_bias(P["df_raw"],P["df_enc"],best_r["model"],LABEL,P["features"]); overall_acc=best_r["test_acc"]
    st.markdown(f'<div class="ok-box">Best model: <b>{best_name}</b> · Overall Test Accuracy: <b>{overall_acc:.1%}</b></div>',unsafe_allow_html=True)

    # Bias table
    st.markdown('<div class="sec-title">Group-Level Performance Gaps</div>',unsafe_allow_html=True)
    if not bias_df.empty:
        btbl="""<table class="stbl"><thead><tr>
          <th>Group</th><th>Value</th><th>N</th>
          <th>Accuracy</th><th>+ve Rate</th><th>TPR</th><th>FPR</th><th>Acc Gap</th>
        </tr></thead><tbody>"""
        for _,row in bias_df.iterrows():
            gap=row["Accuracy"]-overall_acc; clr="#065F46" if gap>=0 else "#991B1B"
            btbl+=f"""<tr><td><b>{row['Group Column']}</b></td><td>{row['Group Value']}</td>
              <td>{row['N']}</td><td>{_badge(row['Accuracy'])}</td>
              <td>{row['Positive Rate']*100:.1f}%</td><td>{row['TPR']*100:.1f}%</td>
              <td>{row['FPR']*100:.1f}%</td>
              <td style='color:{clr};font-weight:700;'>{gap*100:+.1f}%</td></tr>"""
        btbl+="</tbody></table>"; st.markdown(btbl,unsafe_allow_html=True)

    # ── Group accuracy bars + slope charts ───────────────────────────
    if not bias_df.empty:
        for gc in bias_df["Group Column"].unique():
            sub=bias_df[bias_df["Group Column"]==gc].copy()
            b1,b2=st.columns(2)
            with b1:
                fig=go.Figure()
                fig.add_trace(go.Bar(x=sub["Group Value"],y=sub["Accuracy"]*100,name="Accuracy",
                    marker_color=C["primary"],text=[f"{v*100:.1f}%" for v in sub["Accuracy"]],textposition="outside"))
                fig.add_trace(go.Bar(x=sub["Group Value"],y=sub["Positive Rate"]*100,name="+ve Rate",
                    marker_color=C["warning"],text=[f"{v*100:.1f}%" for v in sub["Positive Rate"]],textposition="outside"))
                fig.add_hline(y=overall_acc*100,line_dash="dash",line_color=C["danger"],
                    annotation_text=f"Overall {overall_acc:.1%}",annotation_font=dict(size=10,color=C["danger"]))
                fig.update_layout(**L(barmode="group",title=dict(text=f"{gc} — Accuracy & Positive Rate",font=dict(size=13)),
                    yaxis=dict(range=[0,115],gridcolor=C["grid"]),
                    legend=dict(orientation="h",yanchor="bottom",y=1.02),height=340))
                st.plotly_chart(fig,use_container_width=True)
            with b2:
                fig=go.Figure()
                colours=list(MODEL_CLR.values())
                for i,(_,row) in enumerate(sub.iterrows()):
                    clr=colours[i%len(colours)]
                    fig.add_trace(go.Scatter(x=["TPR","FPR"],y=[row["TPR"]*100,row["FPR"]*100],
                        mode="lines+markers+text",name=row["Group Value"],
                        text=[f"{row['Group Value']} {row['TPR']*100:.1f}%",f"{row['FPR']*100:.1f}%"],
                        textposition=["middle left","middle right"],textfont=dict(size=10),
                        line=dict(color=clr,width=2.5),marker=dict(size=11)))
                fig.update_layout(**L(title=dict(text=f"{gc} — TPR vs FPR Slope Chart",font=dict(size=13)),
                    xaxis=dict(tickfont=dict(size=13)),yaxis=dict(title="%",gridcolor=C["grid"]),
                    showlegend=True,legend=dict(font=dict(size=10)),height=340))
                st.plotly_chart(fig,use_container_width=True)

    # ── Disparate Impact Ratio ────────────────────────────────────────
    st.markdown('<div class="sec-title">📐 Disparate Impact Ratio (DIR)</div>',unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>DIR = Min group +ve rate / Max group +ve rate.</b> EEOC 4/5ths rule: DIR &lt; 0.80 signals potential discriminatory impact.</div>',unsafe_allow_html=True)
    gen_sub=bias_df[bias_df["Group Column"]=="GENDER"] if not bias_df.empty else pd.DataFrame()
    if len(gen_sub)>=2:
        gs=gen_sub.sort_values("Positive Rate",ascending=False)
        maj_r=gs.iloc[0]["Positive Rate"]; min_r=gs.iloc[-1]["Positive Rate"]
        dir_v=min_r/maj_r if maj_r>0 else 1.0
        tpr_gap=abs(gen_sub["TPR"].max()-gen_sub["TPR"].min())
        d1,d2,d3,d4=st.columns(4)
        d1.markdown(f"""<div class="kpi blue"><div class="kpi-label">Majority +ve Rate</div>
          <div class="kpi-val">{maj_r*100:.1f}%</div><div class="kpi-sub">{gs.iloc[0]['Group Value']}</div></div>""",unsafe_allow_html=True)
        d2.markdown(f"""<div class="kpi amber"><div class="kpi-label">Minority +ve Rate</div>
          <div class="kpi-val">{min_r*100:.1f}%</div><div class="kpi-sub">{gs.iloc[-1]['Group Value']}</div></div>""",unsafe_allow_html=True)
        d3.markdown(f"""<div class="kpi {'green' if dir_v>=0.80 else 'red'}"><div class="kpi-label">DIR</div>
          <div class="kpi-val">{dir_v:.3f}</div><div class="kpi-sub">{'✅ Fair' if dir_v>=0.80 else '⚠️ Biased'}</div></div>""",unsafe_allow_html=True)
        d4.markdown(f"""<div class="kpi violet"><div class="kpi-label">TPR Gap (Gender)</div>
          <div class="kpi-val">{tpr_gap*100:.1f}%</div><div class="kpi-sub">Equal Opportunity Δ</div></div>""",unsafe_allow_html=True)

        # ── Gauge ────────────────────────────────────────────────────
        fig=go.Figure(go.Indicator(mode="gauge+number+delta",value=dir_v,
            delta={"reference":0.80,"increasing":{"color":C["success"]},"decreasing":{"color":C["danger"]}},
            gauge={"axis":{"range":[0,1.2],"tickwidth":1},
                   "bar":{"color":C["success"] if dir_v>=0.80 else C["danger"]},
                   "bgcolor":"#F8FAFC","borderwidth":2,"bordercolor":"#E2E8F0",
                   "steps":[{"range":[0,0.8],"color":"#FEE2E2"},
                              {"range":[0.8,1.0],"color":"#D1FAE5"},
                              {"range":[1.0,1.2],"color":"#DBEAFE"}],
                   "threshold":{"line":{"color":C["danger"],"width":3},"thickness":0.8,"value":0.80}},
            title={"text":"Disparate Impact Ratio (0.80 threshold)","font":{"size":14}},
            number={"font":{"size":40,"color":C["text"]}}))
        fig.update_layout(**L(height=300,margin=dict(l=20,r=20,t=60,b=20)))
        st.plotly_chart(fig,use_container_width=True)

    # ── Lollipop chart ────────────────────────────────────────────────
    st.markdown('<div class="sec-title">Lollipop — Per-Group Accuracy vs Baseline</div>',unsafe_allow_html=True)
    if not bias_df.empty:
        ldf=bias_df.copy(); ldf["Label"]=ldf["Group Column"]+": "+ldf["Group Value"]
        ldf=ldf.sort_values("Accuracy",ascending=True)
        fig=go.Figure()
        for _,row in ldf.iterrows():
            clr=C["success"] if row["Accuracy"]>=overall_acc else C["danger"]
            fig.add_trace(go.Scatter(x=[0,row["Accuracy"]*100],y=[row["Label"],row["Label"]],
                mode="lines",line=dict(color=clr,width=2),showlegend=False))
        fig.add_trace(go.Scatter(x=ldf["Accuracy"]*100,y=ldf["Label"],mode="markers+text",
            text=[f"{v*100:.1f}%" for v in ldf["Accuracy"]],textposition="middle right",textfont=dict(size=10),
            marker=dict(size=14,color=ldf["Accuracy"]*100,
                        colorscale=[[0,C["danger"]],[0.6,C["warning"]],[1,C["success"]]],
                        showscale=True,colorbar=dict(title="Acc %",thickness=10),
                        line=dict(width=2,color="#FFF")),showlegend=False))
        fig.add_vline(x=overall_acc*100,line_dash="dash",line_color=C["slate"],
            annotation_text=f"Overall {overall_acc:.1%}",annotation_font=dict(size=11))
        fig.update_layout(**L(title=dict(text="Per-Group Accuracy vs Overall Baseline",font=dict(size=14)),
            xaxis=dict(title="Accuracy (%)",range=[0,115],gridcolor=C["grid"]),
            height=max(350,len(ldf)*40),margin=dict(l=10,r=80,t=50,b=30)))
        st.plotly_chart(fig,use_container_width=True)

    # ── Class imbalance ───────────────────────────────────────────────
    st.markdown('<div class="sec-title">Class Imbalance Analysis</div>',unsafe_allow_html=True)
    vc2=P["df_raw"][LABEL].value_counts(); ir=vc2.min()/vc2.max()
    ci1,ci2,ci3=st.columns(3)
    ci1.markdown(f"""<div class="kpi {'red' if ir<0.4 else 'amber'}"><div class="kpi-label">Imbalance Ratio</div>
      <div class="kpi-val">{ir:.2f}</div><div class="kpi-sub">minority/majority</div></div>""",unsafe_allow_html=True)
    ci2.markdown(f"""<div class="kpi green"><div class="kpi-label">Majority Class</div>
      <div class="kpi-val">{vc2.index[0]} ({vc2.iloc[0]})</div><div class="kpi-sub">{vc2.iloc[0]/vc2.sum():.0%}</div></div>""",unsafe_allow_html=True)
    ci3.markdown(f"""<div class="kpi amber"><div class="kpi-label">Minority Class</div>
      <div class="kpi-val">{vc2.index[-1]} ({vc2.iloc[-1]})</div><div class="kpi-sub">{vc2.iloc[-1]/vc2.sum():.0%}</div></div>""",unsafe_allow_html=True)
    ib=("mild" if ir>0.5 else "moderate" if ir>0.2 else "severe")
    st.markdown(f"""<div class="{'ok-box' if ir>0.5 else 'warn-box'}">Imbalance is <b>{ib}</b> (ratio {ir:.2f}).
      {'No resampling needed.' if ir>0.5 else 'Consider SMOTE or <code>class_weight="balanced"</code>.'}</div>""",unsafe_allow_html=True)

    # ── Mitigation ────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">💡 Bias Mitigation Playbook</div>',unsafe_allow_html=True)
    st.markdown("""<div class="ok-box"><ol style='margin:0;padding-left:20px;line-height:2.1;'>
      <li><b>Re-weighting:</b> <code>class_weight='balanced'</code> in tree/logistic models.</li>
      <li><b>SMOTE oversampling:</b> Synthetically oversample the minority class.</li>
      <li><b>Fairness-aware training:</b> Use <code>Fairlearn</code> / <code>AIF360</code>.</li>
      <li><b>Equalised Odds:</b> Post-hoc calibration for equal TPR/FPR across groups.</li>
      <li><b>Stratified reporting:</b> Always report group-level metrics.</li>
      <li><b>Diverse data collection:</b> Proportional representation of all demographics.</li>
      <li><b>Mandatory clinical review:</b> Escalate high-risk under-represented patients.</li>
    </ol></div>""",unsafe_allow_html=True)
