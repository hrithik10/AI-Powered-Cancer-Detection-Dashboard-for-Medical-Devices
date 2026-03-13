# 🫁 Lung Cancer Detection — ML Intelligence Dashboard

A comprehensive, interactive **Streamlit** dashboard for in-depth lung cancer prediction analysis, covering classification, regression, association rule mining, and demographic bias detection.

---

## 🖼️ Dashboard Tabs

| Tab | What's inside |
|-----|---------------|
| **📊 Data Overview** | Steps 2–4: shape, null checks, label encoding map, chi-square significance test, correlation heatmap, age distribution |
| **🌳 Classification** | Steps 7–10: 7 models benchmarked, accuracy/precision/recall/F1/AUC table, grouped bar, ROC curves, confusion matrices, feature importance, PR curves, CV box plots |
| **📈 Regression** | Logistic Regression, Ridge, SVM, KNN, Naive Bayes — radar chart, signed coefficient chart, full ranking |
| **🔗 Association Rules** | Apriori — lift/support/confidence/leverage/conviction tables, top-20 bar chart, scatter plot, LUNG_CANCER-specific rules |
| **⚖️ Bias Detection** | Gender/age/smoking accuracy gaps, Disparate Impact Ratio, class imbalance analysis, mitigation playbook |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/lung-cancer-dashboard.git
cd lung-cancer-dashboard

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub (include `survey_lung_cancer.csv`)
2. Visit [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo · branch `main` · Main file: `app.py`
4. Click **Deploy**

---

## 📁 Project Structure

```
lung-cancer-dashboard/
├── app.py                     ← Full Streamlit application (single file)
├── requirements.txt           ← Pinned dependencies
├── survey_lung_cancer.csv     ← Dataset
├── .streamlit/
│   └── config.toml           ← Light theme configuration
└── README.md
```

---

## 📊 Models Benchmarked

| Model | Family | Notes |
|-------|--------|-------|
| Decision Tree | Tree | Interpretable, prone to overfitting |
| Random Forest | Ensemble | Best overall accuracy on this dataset |
| Gradient Boosting | Ensemble | Sequential error correction |
| Logistic Regression | Linear | Probabilistic, interpretable coefficients |
| SVM | Kernel | Effective on small datasets |
| KNN | Instance-based | Simple, no training phase |
| Naive Bayes | Probabilistic | Fast, assumes feature independence |

---

## 📦 Dependencies

```
streamlit >= 1.35
scikit-learn >= 1.4
plotly >= 5.20
mlxtend >= 0.23
pandas >= 2.0
numpy >= 1.24
scipy >= 1.11
```

---

## 🔑 Key Findings (on included dataset)

- **Random Forest** achieves the highest test accuracy (~91.9%)
- **ALLERGY, ALCOHOL CONSUMING, SWALLOWING DIFFICULTY** are top correlated features
- Class imbalance: 87% YES / 13% NO — models benefit from `class_weight='balanced'`
- Disparate Impact Ratio for gender is within the 0.80 fair threshold on this dataset

---

## ⚠️ Notes

- All graphs are interactive (Plotly) — hover, zoom, download as PNG
- Sidebar sliders control tree hyperparameters and ARM thresholds in real time
- Upload your own CSV via the sidebar to analyse a different dataset
