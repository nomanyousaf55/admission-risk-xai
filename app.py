"""
Interactive Decision-Support Dashboard (Chapter 3.4 — Dashboard).

Streamlit application that presents, for a selected or newly-entered young
person: the predicted admission-risk probability and Low/Medium/High
category; the SHAP explanation of that prediction (as unmet-need signals);
the resulting community-care recommendations; and cohort-level analytics
including risk distribution, global feature importance and model-performance
metrics.

Run from the implementation folder:
    streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from config import (  # noqa: E402
    BEST_MODEL_PATH,
    METADATA_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    risk_category,
)
from preprocessing import load_data, split_X_y  # noqa: E402
from explainability import build_explainer, global_importance, local_explanation  # noqa: E402
from recommendation_engine import recommend  # noqa: E402

st.set_page_config(
    page_title="Admission Risk Decision Support",
    page_icon="🏥",
    layout="wide",
)

RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BEST_MODEL_PATH)
    meta = joblib.load(METADATA_PATH)
    return model, meta


@st.cache_data
def load_dataset():
    df = load_data()
    return df


@st.cache_resource
def get_explainer():
    model, _ = load_artifacts()
    df = load_dataset()
    X, _ = split_X_y(df)
    expl, names = build_explainer(model, X.sample(150, random_state=1))
    return expl


@st.cache_data
def get_global_importance():
    model, _ = load_artifacts()
    df = load_dataset()
    X, _ = split_X_y(df)
    expl = get_explainer()
    return global_importance(expl, model, X.sample(200, random_state=2))


def predict_proba_row(model, row_df):
    return float(model.predict_proba(row_df)[:, 1][0])


# ----------------------------------------------------------------------
# Load everything
# ----------------------------------------------------------------------
try:
    model, meta = load_artifacts()
except FileNotFoundError:
    st.error("Trained model not found. Run `python src/model_training.py` first.")
    st.stop()

df = load_dataset()
X, y = split_X_y(df)

st.title("🏥 Explainable Hospital Admission Risk & Community-Care Decision Support")
st.caption(
    "Young people with autism and/or a learning disability — predict avoidable "
    "admission risk, explain the unmet needs behind it, and recommend "
    "preventative community-care actions. Decision-support aid for clinicians "
    "— not an autonomous decision-maker."
)

tab_patient, tab_cohort, tab_perf = st.tabs(
    ["👤 Individual Assessment", "📊 Cohort Analytics", "📈 Model Performance"]
)

# ======================================================================
# TAB 1 — Individual assessment
# ======================================================================
with tab_patient:
    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("Patient input")
        mode = st.radio(
            "Choose input mode", ["Pick from dataset", "Enter manually"],
            horizontal=True,
        )

        if mode == "Pick from dataset":
            idx = st.number_input(
                "Record index", min_value=0, max_value=len(X) - 1, value=0, step=1
            )
            row = X.iloc[[int(idx)]].copy()
            disp = row.T.reset_index()
            disp.columns = ["Feature", "Value"]
            disp["Feature"] = disp["Feature"].map(FEATURE_LABELS).fillna(disp["Feature"])
            disp["Value"] = disp["Value"].astype(str)
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            vals = {}
            for f in NUMERIC_FEATURES:
                col = X[f]
                vals[f] = st.slider(
                    FEATURE_LABELS.get(f, f),
                    int(col.min()), int(col.max()), int(col.median()),
                )
            for f in CATEGORICAL_FEATURES:
                opts = sorted(X[f].unique().tolist())
                vals[f] = st.selectbox(FEATURE_LABELS.get(f, f), opts)
            row = pd.DataFrame([vals])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        assess = st.button("Assess risk", type="primary")

    with right:
        if assess or mode == "Pick from dataset":
            prob = predict_proba_row(model, row)
            cat = risk_category(prob)

            st.subheader("Predicted admission risk")
            m1, m2 = st.columns(2)
            m1.metric("Risk probability", f"{prob*100:.1f}%")
            m2.markdown(
                f"<div style='padding:14px;border-radius:8px;background:"
                f"{RISK_COLORS[cat]};color:white;text-align:center;font-size:22px;"
                f"font-weight:700'>{cat} risk</div>",
                unsafe_allow_html=True,
            )

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": RISK_COLORS[cat]},
                    "steps": [
                        {"range": [0, 30], "color": "#eafaf1"},
                        {"range": [30, 60], "color": "#fef5e7"},
                        {"range": [60, 100], "color": "#fdedec"},
                    ],
                },
            ))
            gauge.update_layout(height=230, margin=dict(t=10, b=10))
            st.plotly_chart(gauge, use_container_width=True)

            # ---- SHAP local explanation ----
            st.subheader("Why? — unmet-need signals (SHAP)")
            with st.spinner("Computing explanation..."):
                expl = get_explainer()
                le = local_explanation(expl, model, row)
            top = le.head(7).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top["shap_value"],
                y=top["label"],
                orientation="h",
                marker_color=["#e74c3c" if v > 0 else "#3498db"
                              for v in top["shap_value"]],
            ))
            fig.update_layout(
                height=320, margin=dict(t=10, b=10),
                xaxis_title="← lowers risk    contribution    raises risk →",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---- Recommendations ----
            st.subheader("Recommended community-care actions")
            recs = recommend(row.iloc[0].to_dict(), le)
            if recs:
                for r in recs:
                    st.markdown(
                        f"**• {r['driver']}**  \n"
                        f"&nbsp;&nbsp;&nbsp;→ {r['action']}"
                    )
            else:
                st.info("No high-impact unmet-need drivers triggered for this "
                        "individual.")

# ======================================================================
# TAB 2 — Cohort analytics
# ======================================================================
with tab_cohort:
    st.subheader("Cohort risk overview")
    probs = model.predict_proba(X)[:, 1]
    cats = pd.Series([risk_category(p) for p in probs])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", len(X))
    c2.metric("High risk", int((cats == "High").sum()))
    c3.metric("Medium risk", int((cats == "Medium").sum()))
    c4.metric("Actual admission rate", f"{y.mean()*100:.1f}%")

    cc1, cc2 = st.columns(2)
    with cc1:
        dist = cats.value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig = px.bar(
            x=dist.index, y=dist.values,
            color=dist.index, color_discrete_map=RISK_COLORS,
            labels={"x": "Risk category", "y": "Patients"},
            title="Risk category distribution",
        )
        fig.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        fig = px.histogram(
            probs, nbins=30, title="Predicted risk probability distribution",
            labels={"value": "Risk probability"},
        )
        fig.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Global feature importance (mean |SHAP|)")
    gi = get_global_importance().head(10).iloc[::-1]
    fig = px.bar(
        gi, x="importance", y="label", orientation="h",
        labels={"importance": "Mean |SHAP| value", "label": ""},
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# TAB 3 — Model performance
# ======================================================================
with tab_perf:
    st.subheader(f"Best model: {meta['best_name']}")
    results = pd.DataFrame(meta["results"])
    show_cols = ["Model", "ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"]
    st.dataframe(
        results[show_cols].style.highlight_max(
            subset=["ROC-AUC", "PR-AUC", "Recall", "F1"], color="#d5f5e3"
        ),
        use_container_width=True,
    )

    p1, p2 = st.columns(2)
    with p1:
        roc = meta["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"],
                                 name="ROC", line=dict(color="#2980b9")))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance",
                                 line=dict(dash="dash", color="grey")))
        fig.update_layout(title="ROC curve", height=340,
                          xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        pr = meta["pr_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"],
                                 name="PR", line=dict(color="#c0392b")))
        fig.update_layout(title="Precision–Recall curve", height=340,
                          xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    cm = np.array(meta["confusion_matrix"])
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
        title="Confusion matrix (best model)",
    )
    fig.update_layout(height=340)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fairness audit — recall by subgroup")
    st.caption("Checks whether the model misses high-risk cases unequally "
               "across demographic groups (Chapter 3.3).")
    st.dataframe(pd.DataFrame(meta["fairness"]), use_container_width=True)
