"""
Train & evaluate the five models on the REAL public dataset
(UCI Diabetes 130-US Hospitals — early-readmission target).

Mirrors model_training.py exactly (same pipeline: preprocess -> SMOTE on
train folds only -> classifier, GridSearchCV 5-fold, PR-AUC scoring, same
imbalanced-data metrics and fairness audit) but on real data. Saves a
separate set of artefacts so the synthetic run is preserved.

Run:  python train_real.py
"""
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    precision_score, f1_score, confusion_matrix, roc_curve,
    precision_recall_curve,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from config import RANDOM_STATE, MODELS_DIR, OUTPUTS_DIR
from real_data_loader import NUMERIC, CATEGORICAL, TARGET, OUT as REAL_CSV

warnings.filterwarnings("ignore")

BEST_MODEL_PATH = MODELS_DIR / "best_model_real.joblib"
METADATA_PATH = MODELS_DIR / "metadata_real.joblib"
RESULTS_PATH = OUTPUTS_DIR / "model_comparison_real.csv"

# use a stratified subsample for tractable GridSearch on 100k rows
SUBSAMPLE = 25000


def build_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC),
                              ("cat", categorical, CATEGORICAL)])


def model_zoo():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"clf__C": [0.1, 1.0]}),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"clf__n_estimators": [300], "clf__max_depth": [None, 15]}),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"clf__C": [1.0], "clf__kernel": ["rbf"]}),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"clf__n_estimators": [200], "clf__learning_rate": [0.1]}),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                          use_label_encoder=False, verbosity=0, n_jobs=-1),
            {"clf__n_estimators": [300], "clf__max_depth": [4, 6],
             "clf__learning_rate": [0.1]}),
    }


def build_pipeline(est):
    return ImbPipeline([("preprocessor", build_preprocessor()),
                        ("smote", SMOTE(random_state=RANDOM_STATE)),
                        ("clf", est)])


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba),
        "Recall": recall_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred),
        "_proba": proba, "_pred": pred,
    }


def fairness_audit(X_test, y_test, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    df = X_test.copy()
    df["_y"] = y_test.values
    df["_pred"] = pred
    df["age_band"] = pd.cut(df["age_years"], bins=[0, 40, 65, 100],
                            labels=["<40", "40-65", "65+"])
    rows = []
    for attr in ["gender", "race", "age_band"]:
        for group, g in df.groupby(attr, observed=True):
            pos = g[g["_y"] == 1]
            recall = (pos["_pred"] == 1).mean() if len(pos) else np.nan
            rows.append({"Attribute": attr, "Group": str(group), "N": len(g),
                         "Positives": len(pos),
                         "Recall": round(recall, 3) if not np.isnan(recall) else None,
                         "FNR": round(1 - recall, 3) if not np.isnan(recall) else None})
    return pd.DataFrame(rows)


def get_feature_names(preprocessor):
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    return list(NUMERIC) + list(ohe.get_feature_names_out(CATEGORICAL))


def main():
    print("Loading REAL dataset ...")
    df = pd.read_csv(REAL_CSV)
    if SUBSAMPLE and len(df) > SUBSAMPLE:
        frac = SUBSAMPLE / len(df)
        df = df.groupby(TARGET, group_keys=False)[df.columns].apply(
            lambda g: g.sample(frac=frac, random_state=RANDOM_STATE)
        ).reset_index(drop=True)
        print(f"  stratified subsample: {len(df):,} rows")
    X = df[NUMERIC + CATEGORICAL].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print(f"  train={len(X_train):,}  test={len(X_test):,}  "
          f"positive rate={y.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = [], {}
    for name, (est, grid) in model_zoo().items():
        print(f"\nTraining {name} ...")
        search = GridSearchCV(build_pipeline(est), grid,
                              scoring="average_precision", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        m = evaluate(search.best_estimator_, X_test, y_test)
        fitted[name] = search.best_estimator_
        results.append({"Model": name, "ROC-AUC": round(m["ROC-AUC"], 4),
                        "PR-AUC": round(m["PR-AUC"], 4), "Recall": round(m["Recall"], 4),
                        "Precision": round(m["Precision"], 4), "F1": round(m["F1"], 4)})
        print(f"  ROC-AUC={m['ROC-AUC']:.3f}  PR-AUC={m['PR-AUC']:.3f}  "
              f"Recall={m['Recall']:.3f}")

    res = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)
    res.to_csv(RESULTS_PATH, index=False)
    print("\n===== Model comparison on REAL data (by PR-AUC) =====")
    print(res.to_string(index=False))

    best_name = res.iloc[0]["Model"]
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")

    bm = evaluate(best, X_test, y_test)
    proba = bm["_proba"]
    cm = confusion_matrix(y_test, bm["_pred"])
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    fair = fairness_audit(X_test, y_test, proba)
    fair.to_csv(OUTPUTS_DIR / "fairness_audit_real.csv", index=False)
    print("\n===== Fairness audit (real) =====")
    print(fair.to_string(index=False))

    joblib.dump(best, BEST_MODEL_PATH)
    joblib.dump({
        "best_name": best_name, "results": res.to_dict(orient="records"),
        "feature_names": get_feature_names(best.named_steps["preprocessor"]),
        "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
        "fairness": fair.to_dict(orient="records"),
        "test_admission_rate": float(y_test.mean()),
        "dataset": "UCI Diabetes 130-US Hospitals (early readmission)",
        "n_total": int(len(df)),
    }, METADATA_PATH)
    print(f"\nSaved -> {BEST_MODEL_PATH}\nSaved -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
