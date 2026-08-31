"""
Explainable AI layer (Chapter 3.4 — SHAP Explainability).

Wraps SHAP so that the trained pipeline can be explained at two levels:
  - global: which features drive admission risk across the population;
  - local: why one particular young person is flagged.

SHAP operates on the preprocessed (numeric) feature space; helper functions
map the attributions back to readable feature names, and aggregate one-hot
columns back to their original categorical feature so each driver reads as an
unmet-need signal.
"""
import numpy as np
import pandas as pd
import shap

from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_LABELS


def _transform(model, X):
    """Apply the fitted preprocessor, return dense array + feature names."""
    pre = model.named_steps["preprocessor"]
    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    names = list(NUMERIC_FEATURES) + list(
        ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return np.asarray(Xt), names


def build_explainer(model, X_background):
    """Create a SHAP explainer appropriate to the fitted classifier."""
    clf = model.named_steps["clf"]
    Xt_bg, names = _transform(model, X_background)

    tree_types = ("RandomForest", "GradientBoosting", "XGB")
    if any(t in type(clf).__name__ for t in tree_types):
        explainer = shap.TreeExplainer(clf)
    else:
        # linear / kernel models — use a small background sample
        bg = shap.sample(Xt_bg, min(100, len(Xt_bg)), random_state=0)
        explainer = shap.Explainer(clf.predict_proba, bg)
    return explainer, names


def _shap_matrix(explainer, Xt):
    """Return a 2D SHAP array for the positive (admission) class."""
    sv = explainer.shap_values(Xt) if hasattr(explainer, "shap_values") \
        else explainer(Xt).values
    sv = np.asarray(sv)
    if sv.ndim == 3:            # (samples, features, classes)
        sv = sv[:, :, 1]
    return sv


def global_importance(explainer, model, X):
    """Mean |SHAP| per feature, aggregated to original feature names."""
    Xt, names = _transform(model, X)
    sv = _shap_matrix(explainer, Xt)
    mean_abs = np.abs(sv).mean(axis=0)

    agg = {}
    for name, val in zip(names, mean_abs):
        base = name
        for cat in CATEGORICAL_FEATURES:
            if name.startswith(cat + "_"):
                base = cat
                break
        agg[base] = agg.get(base, 0.0) + float(val)

    out = pd.DataFrame(
        {"feature": list(agg.keys()), "importance": list(agg.values())}
    )
    out["label"] = out["feature"].map(FEATURE_LABELS).fillna(out["feature"])
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def local_explanation(explainer, model, x_row: pd.DataFrame):
    """Signed SHAP contribution for a single individual, per original feature."""
    Xt, names = _transform(model, x_row)
    sv = _shap_matrix(explainer, Xt)[0]

    agg = {}
    for name, val in zip(names, sv):
        base = name
        for cat in CATEGORICAL_FEATURES:
            if name.startswith(cat + "_"):
                base = cat
                break
        agg[base] = agg.get(base, 0.0) + float(val)

    out = pd.DataFrame(
        {"feature": list(agg.keys()), "shap_value": list(agg.values())}
    )
    out["label"] = out["feature"].map(FEATURE_LABELS).fillna(out["feature"])
    out["abs"] = out["shap_value"].abs()
    return out.sort_values("abs", ascending=False).reset_index(drop=True)
