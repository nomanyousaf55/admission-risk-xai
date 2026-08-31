"""
Real-data loader (public dataset) — Chapter 3.1 / Chapter 4.2.

Loads the UCI "Diabetes 130-US Hospitals (1999-2008)" dataset — 101,766 real
hospital encounters from 130 US hospitals (Strack et al., 2014), a widely-used,
peer-reviewed public benchmark for hospital *readmission* prediction. Early
readmission (readmitted within 30 days) is used as the target, serving as a
real-world proxy for avoidable admission risk.

Source: UCI Machine Learning Repository, dataset 296.
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
Licence: CC BY 4.0.

This module selects clinically meaningful features that map onto the feature
groups used in the methodology (demographic, clinical history, service-contact
/ utilisation, care-process), cleans them, and writes a tidy CSV that the rest
of the pipeline consumes exactly like the synthetic data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "real" / "dataset_diabetes" / "diabetic_data.csv"
OUT = Path(__file__).resolve().parent.parent / "data" / "real_admission_dataset.csv"

TARGET = "readmitted_early"

# --- feature selection (maps onto the four methodology groups) ---
NUMERIC = [
    "age_years",              # demographic (midpoint of age band)
    "time_in_hospital",      # clinical history (length of stay)
    "num_medications",       # clinical history
    "num_lab_procedures",    # clinical history
    "num_procedures",        # clinical history
    "number_diagnoses",      # clinical history
    "number_inpatient",      # prior utilisation (previous admissions)
    "number_emergency",      # prior utilisation (crises proxy)
    "number_outpatient",     # service-engagement (community contact proxy)
]
CATEGORICAL = [
    "gender",                # demographic
    "race",                  # demographic
    "admission_type",        # care-process
    "A1Cresult",             # clinical
    "insulin",               # clinical (medication)
    "change",                # care-process (medication changed)
    "diabetesMed",           # care-process
]

_AGE_MID = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}
_ADM_TYPE = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Not Available", 6: "NULL", 7: "Trauma Centre", 8: "Not Mapped",
}


def build(save=True) -> pd.DataFrame:
    df = pd.read_csv(RAW)

    # target: early readmission (<30 days) = 1, else 0  (minority class)
    df[TARGET] = (df["readmitted"] == "<30").astype(int)

    # --- clean / derive features ---
    df["age_years"] = df["age"].map(_AGE_MID)
    df["admission_type"] = df["admission_type_id"].map(_ADM_TYPE).fillna("Unknown")

    # replace '?' with NaN in categoricals we keep
    for col in ["race", "A1Cresult", "insulin", "gender"]:
        df[col] = df[col].replace("?", np.nan)
    # A1Cresult uses 'None' as a genuine category meaning not measured
    df["A1Cresult"] = df["A1Cresult"].fillna("Not measured").replace("None", "Not measured")

    # drop rows with unknown/invalid gender
    df = df[df["gender"].isin(["Male", "Female"])].copy()

    keep = NUMERIC + CATEGORICAL + [TARGET]
    tidy = df[keep].copy()

    if save:
        tidy.to_csv(OUT, index=False)
    return tidy


if __name__ == "__main__":
    d = build()
    print(f"Built real dataset -> {OUT}")
    print(f"Rows: {len(d):,}  Features: {len(NUMERIC) + len(CATEGORICAL)}")
    print(f"Early-readmission (positive) rate: {d[TARGET].mean():.3f}")
    print(d.head())
