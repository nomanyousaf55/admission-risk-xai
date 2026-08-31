"""
Central configuration for the Explainable AI Hospital Admission Risk
Prediction and Community-Care Decision Support System.

All paths, feature definitions, model settings and the SHAP-driver ->
community-care action rules live here so the rest of the pipeline stays clean.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "admission_risk_dataset.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.joblib"
RESULTS_PATH = OUTPUTS_DIR / "model_comparison.csv"

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# Dataset design  (Chapter 3.1 — four feature groups)
# ----------------------------------------------------------------------
N_SAMPLES = 3000
# Admissions are the minority class (real-world imbalance we must handle)
ADMISSION_RATE = 0.18

TARGET = "admitted"

# Group 1 — Demographic
# Group 2 — Clinical history
# Group 3 — Service-engagement
# Group 4 — Social / environmental
NUMERIC_FEATURES = [
    "age",
    "num_previous_crises",
    "num_previous_admissions",
    "days_since_last_contact",
    "community_contacts_last_year",
    "medication_changes_last_year",
]
CATEGORICAL_FEATURES = [
    "sex",
    "ethnicity",
    "diagnosis",
    "care_plan_reviewed",
    "placement_stability",
    "family_support_level",
    "respite_available",
    "education_engagement",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Human-readable labels for the dashboard / SHAP plots
FEATURE_LABELS = {
    "age": "Age",
    "num_previous_crises": "Previous crises",
    "num_previous_admissions": "Previous admissions",
    "days_since_last_contact": "Days since last service contact",
    "community_contacts_last_year": "Community contacts (last year)",
    "medication_changes_last_year": "Medication changes (last year)",
    "sex": "Sex",
    "ethnicity": "Ethnicity",
    "diagnosis": "Diagnosis profile",
    "care_plan_reviewed": "Care plan reviewed",
    "placement_stability": "Placement stability",
    "family_support_level": "Family / carer support",
    "respite_available": "Respite available",
    "education_engagement": "Education / day-activity engagement",
}

# ----------------------------------------------------------------------
# Risk categorisation thresholds (Low / Medium / High)
# ----------------------------------------------------------------------
RISK_THRESHOLDS = {"medium": 0.30, "high": 0.60}


def risk_category(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["high"]:
        return "High"
    if prob >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


# ----------------------------------------------------------------------
# Community-Care Recommendation Engine  (Chapter 3.4 — Table 3.1)
# Each rule maps a risk driver (unmet need) to a preventative action.
# `trigger(row)` decides whether the driver is active for an individual.
# ----------------------------------------------------------------------
RECOMMENDATION_RULES = [
    {
        "driver_feature": "num_previous_crises",
        "driver_label": "Recent crisis / behavioural escalation",
        "action": "Activate crisis / intensive-support team early",
        "trigger": lambda r: r.get("num_previous_crises", 0) >= 2,
    },
    {
        "driver_feature": "community_contacts_last_year",
        "driver_label": "Low community-support engagement",
        "action": "Increase community service contact and outreach",
        "trigger": lambda r: r.get("community_contacts_last_year", 99) <= 4,
    },
    {
        "driver_feature": "placement_stability",
        "driver_label": "Unstable placement or housing",
        "action": "Review placement; arrange supported-living input",
        "trigger": lambda r: str(r.get("placement_stability", "")) == "Unstable",
    },
    {
        "driver_feature": "family_support_level",
        "driver_label": "Limited family / carer support",
        "action": "Provide carer support, respite and education",
        "trigger": lambda r: str(r.get("family_support_level", "")) == "Low",
    },
    {
        "driver_feature": "days_since_last_contact",
        "driver_label": "Long gap since last service contact",
        "action": "Schedule proactive review / care-coordination visit",
        "trigger": lambda r: r.get("days_since_last_contact", 0) >= 120,
    },
]
