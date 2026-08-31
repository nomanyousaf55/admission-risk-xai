# Explainable AI — Hospital Admission Risk & Community-Care Decision Support System

Predicts the risk of avoidable hospital admission for young people with autism
and/or a learning disability, explains **why** each individual is at risk using
SHAP, and recommends preventative community-care actions.

> **The models are already trained and included in this package.**
> You do **not** need to train anything — just run the dashboard.

---

## Quick start (Windows)

1. Make sure **Python 3.11 or newer** is installed
   (download from <https://www.python.org/downloads/> — tick *“Add Python to PATH”*).
2. Double-click **`START_DASHBOARD.bat`**.
3. The first run installs the required packages (a few minutes, once only).
4. The dashboard opens automatically at <http://localhost:8501>.

Leave the black console window open while you use the dashboard; close it when
you are finished.

## Quick start (macOS / Linux, or manual)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Using the dashboard

The interface has three tabs.

### 1. Individual Assessment
Select a record from the dataset (or enter details manually) and press
**Assess risk**. You get:

* the **predicted admission-risk probability** and a Low / Medium / High category,
* a **SHAP explanation** showing which factors push the risk up (red) or down (blue),
* the **recommended community-care actions** for that individual.

*Try record index **2382** — a high-risk case at 99.4% that triggers four
recommendations.*

### 2. Cohort Analytics
Risk distribution across the whole population and the global feature importance —
useful for service-level planning.

### 3. Model Performance
The comparison of all five models, ROC and precision–recall curves, the confusion
matrix, and the subgroup fairness audit.

---

## What the system does

| Stage | Description |
|-------|-------------|
| Preprocessing | Missing-value imputation, one-hot encoding, feature scaling |
| Class imbalance | SMOTE, applied **only** to training folds (no data leakage) |
| Models | Logistic Regression, Random Forest, SVM, Gradient Boosting, XGBoost |
| Tuning | GridSearchCV with stratified 5-fold cross-validation, scored on PR-AUC |
| Explainability | SHAP — global feature importance and per-individual explanations |
| Recommendations | Rule-based engine mapping each risk driver to a preventative action |
| Interface | Streamlit dashboard |

**Best model:** Logistic Regression — ROC-AUC **0.873**, recall **0.802** on the
population-specific dataset; ROC-AUC **0.622** on a real 101,763-record public
benchmark (consistent with published readmission studies).

---

## Project structure

```
implementation/
├── START_DASHBOARD.bat        one-click launcher (Windows)
├── app.py                     Streamlit dashboard
├── requirements.txt           pinned dependency versions
├── data/
│   ├── admission_risk_dataset.csv       population-specific dataset (3,000 records)
│   └── real_admission_dataset.csv       real UCI benchmark (101,763 records)
├── models/                    trained models — ready to use, no training needed
│   ├── best_model.joblib
│   ├── metadata.joblib
│   ├── best_model_real.joblib
│   └── metadata_real.joblib
├── outputs/                   result tables and figures
└── src/
    ├── config.py                    settings, features, recommendation rules
    ├── data_generator.py            generates the synthetic dataset
    ├── real_data_loader.py          prepares the real UCI dataset
    ├── preprocessing.py             transformation pipeline
    ├── model_training.py            training / tuning / evaluation
    ├── train_real.py                same pipeline on the real dataset
    ├── explainability.py            SHAP layer
    └── recommendation_engine.py     community-care recommendations
```

---

## Optional — retraining from scratch

Only needed if you change the data or the model configuration.

```bash
cd src
python data_generator.py     # regenerate the synthetic dataset
python model_training.py     # retrain, retune and re-evaluate (~5 minutes)
python train_real.py         # same on the real dataset (~15 minutes)
```

A fixed random seed (42) is used throughout, so results are exactly reproducible.

---

## Troubleshooting

**“Python was not found”** — install Python and make sure *Add Python to PATH*
was ticked, then restart the launcher.

**Model fails to load** — the bundled models were saved with the exact library
versions pinned in `requirements.txt`. Install those versions rather than the
latest, or retrain with `python src/model_training.py`.

**Port already in use** — run `streamlit run app.py --server.port 8502`.

**The SHAP explanation takes a few seconds** — this is expected. The best model
is logistic regression, which uses a permutation-based explainer.

---

## Data sources

* **Population-specific dataset** — realistically simulated, with feature
  distributions and effect directions derived from CQC (2020) *Out of Sight – Who
  Cares?* and NHS England (2015) *Building the Right Support*. No real individual
  is represented, so there are no data-protection implications.
* **Real benchmark** — *Diabetes 130-US Hospitals (1999–2008)*, UCI Machine
  Learning Repository dataset 296 (Strack et al., 2014), CC BY 4.0.
  <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>

## Important note

This system is a **decision-support aid for qualified professionals**, not an
autonomous decision-maker. It highlights rising risk and suggests preventative
community actions; all care decisions remain with the clinical team.
