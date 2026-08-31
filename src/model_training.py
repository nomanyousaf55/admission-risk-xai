"""
Model development, tuning and evaluation
(Chapter 3.2 — Model Development, Chapter 3.3 — Evaluation).

Trains and compares five supervised classifiers:
    Logistic Regression (baseline), Random Forest, SVM,
    Gradient Boosting, XGBoost.

Each is wrapped in an imbalanced-learn Pipeline
    preprocessor -> SMOTE (train folds only) -> classifier
and tuned with GridSearchCV (stratified 5-fold, recall-oriented scoring).

Evaluation on a held-out stratified test set uses metrics appropriate to
imbalanced clinical data: ROC-AUC, PR-AUC, recall, precision, F1, plus the
confusion matrix. A subgroup fairness audit (recall by sex / ethnicity /
age band) is also produced. The best model (by PR-AUC) is serialised with
Joblib for use by the dashboard.
"""
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from config import (
    RANDOM_STATE,
    BEST_MODEL_PATH,
    METADATA_PATH,
    RESULTS_PATH,
    OUTPUTS_DIR,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from preprocessing import load_data, split_X_y, build_preprocessor, get_feature_names

warnings.filterwarnings("ignore")


def model_zoo():
    """Return (name -> (estimator, param_grid)) for the five algorithms."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10, 20]},
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"clf__C": [0.5, 1.0, 10.0], "clf__kernel": ["rbf"]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"clf__n_estimators": [200, 300], "clf__learning_rate": [0.05, 0.1]},
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
    }


def build_pipeline(estimator):
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", estimator),
        ]
    )


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba),
        "Recall": recall_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred),
        "_proba": proba,
        "_pred": pred,
    }


def fairness_audit(X_test, y_test, proba, threshold=0.5):
    """Recall / false-negative rate by subgroup (Chapter 3.3)."""
    pred = (proba >= threshold).astype(int)
    df = X_test.copy()
    df["_y"] = y_test.values
    df["_pred"] = pred
    df["age_band"] = pd.cut(
        df["age"], bins=[11, 15, 18, 25], labels=["12-15", "16-18", "19-25"]
    )
    rows = []
    for attr in ["sex", "ethnicity", "age_band"]:
        for group, g in df.groupby(attr, observed=True):
            pos = g[g["_y"] == 1]
            recall = (pos["_pred"] == 1).mean() if len(pos) else np.nan
            fnr = 1 - recall if not np.isnan(recall) else np.nan
            rows.append(
                {
                    "Attribute": attr,
                    "Group": str(group),
                    "N": len(g),
                    "Positives": len(pos),
                    "Recall": round(recall, 3) if not np.isnan(recall) else None,
                    "FNR": round(fnr, 3) if not np.isnan(fnr) else None,
                }
            )
    return pd.DataFrame(rows)


def main():
    print("Loading data ...")
    df = load_data()
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  train={len(X_train)}  test={len(X_test)}  "
          f"admission rate={y.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = [], {}

    for name, (estimator, grid) in model_zoo().items():
        print(f"\nTraining {name} ...")
        pipe = build_pipeline(estimator)
        search = GridSearchCV(
            pipe, grid, scoring="average_precision", cv=cv, n_jobs=-1
        )
        search.fit(X_train, y_train)
        metrics = evaluate(search.best_estimator_, X_test, y_test)
        fitted[name] = search.best_estimator_
        results.append(
            {
                "Model": name,
                "ROC-AUC": round(metrics["ROC-AUC"], 4),
                "PR-AUC": round(metrics["PR-AUC"], 4),
                "Recall": round(metrics["Recall"], 4),
                "Precision": round(metrics["Precision"], 4),
                "F1": round(metrics["F1"], 4),
                "Best params": search.best_params_,
            }
        )
        print(f"  ROC-AUC={metrics['ROC-AUC']:.3f}  "
              f"PR-AUC={metrics['PR-AUC']:.3f}  Recall={metrics['Recall']:.3f}")

    results_df = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)
    results_df.to_csv(RESULTS_PATH, index=False)
    print("\n===== Model comparison (sorted by PR-AUC) =====")
    print(results_df[["Model", "ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"]]
          .to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # ---- artefacts for evaluation curves & confusion matrix ----
    best_metrics = evaluate(best_model, X_test, y_test)
    proba = best_metrics["_proba"]
    cm = confusion_matrix(y_test, best_metrics["_pred"])
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)

    # ---- fairness audit ----
    fair = fairness_audit(X_test, y_test, proba)
    fair.to_csv(OUTPUTS_DIR / "fairness_audit.csv", index=False)
    print("\n===== Fairness audit (recall by subgroup) =====")
    print(fair.to_string(index=False))

    # ---- persist best model + metadata ----
    joblib.dump(best_model, BEST_MODEL_PATH)
    preprocessor = best_model.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)
    joblib.dump(
        {
            "best_name": best_name,
            "results": results_df.to_dict(orient="records"),
            "feature_names": feature_names,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "confusion_matrix": cm.tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
            "fairness": fair.to_dict(orient="records"),
            "test_admission_rate": float(y_test.mean()),
        },
        METADATA_PATH,
    )
    print(f"\nSaved best model -> {BEST_MODEL_PATH}")
    print(f"Saved metadata   -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
