"""
Data preprocessing pipeline (Chapter 3.1 — Data Preprocessing).

Builds a scikit-learn ColumnTransformer that:
  - imputes missing values (median for numeric, mode for categorical),
  - one-hot encodes categorical features,
  - standardises numeric features (needed for LR and SVM).

SMOTE is applied ONLY inside the training pipeline (see model_training.py)
using an imbalanced-learn Pipeline, so no synthetic samples ever leak into
validation or test data.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, DATASET_PATH


def load_data(path=DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_X_y(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer):
    """Readable output feature names after one-hot encoding."""
    num = list(NUMERIC_FEATURES)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return num + cat
