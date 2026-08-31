"""
Synthetic dataset generator (Chapter 3.1 — Data Collection).

Generates a realistically simulated dataset for young people with autism
and/or a learning disability. Feature distributions and the relationship
between the risk drivers and admission are designed so that the well-evidenced
drivers of avoidable admission (recent crises, low community engagement,
unstable placement, limited carer support, long gaps in service contact)
genuinely increase admission risk. Admissions are kept as the minority class
to preserve real-world imbalance (see config.ADMISSION_RATE).

The data are fully synthetic — no real individual is represented, which
avoids information-governance and ethical barriers (Chapter 3.4, Ethics).
"""
import numpy as np
import pandas as pd

from config import (
    N_SAMPLES,
    ADMISSION_RATE,
    RANDOM_STATE,
    TARGET,
    DATASET_PATH,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n=N_SAMPLES, seed=RANDOM_STATE, save=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---------------- Group 1: Demographic ----------------
    age = rng.integers(12, 26, size=n)  # young people 12-25
    sex = rng.choice(["Male", "Female"], size=n, p=[0.62, 0.38])
    ethnicity = rng.choice(
        ["White", "Asian", "Black", "Mixed", "Other"],
        size=n, p=[0.72, 0.11, 0.08, 0.06, 0.03],
    )
    diagnosis = rng.choice(
        ["Autism", "Learning disability", "Autism + LD"],
        size=n, p=[0.42, 0.30, 0.28],
    )

    # ---------------- Group 2: Clinical history ----------------
    num_previous_crises = rng.poisson(1.1, size=n)
    num_previous_admissions = rng.poisson(0.4, size=n)
    medication_changes_last_year = rng.poisson(1.0, size=n)

    # ---------------- Group 3: Service-engagement ----------------
    days_since_last_contact = rng.integers(1, 300, size=n)
    community_contacts_last_year = rng.poisson(9, size=n)
    care_plan_reviewed = rng.choice(["Yes", "No"], size=n, p=[0.65, 0.35])

    # ---------------- Group 4: Social / environmental ----------------
    placement_stability = rng.choice(
        ["Stable", "At risk", "Unstable"], size=n, p=[0.55, 0.28, 0.17]
    )
    family_support_level = rng.choice(
        ["High", "Medium", "Low"], size=n, p=[0.34, 0.42, 0.24]
    )
    respite_available = rng.choice(["Yes", "No"], size=n, p=[0.45, 0.55])
    education_engagement = rng.choice(
        ["Engaged", "Partial", "Not engaged"], size=n, p=[0.5, 0.32, 0.18]
    )

    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "ethnicity": ethnicity,
            "diagnosis": diagnosis,
            "num_previous_crises": num_previous_crises,
            "num_previous_admissions": num_previous_admissions,
            "medication_changes_last_year": medication_changes_last_year,
            "days_since_last_contact": days_since_last_contact,
            "community_contacts_last_year": community_contacts_last_year,
            "care_plan_reviewed": care_plan_reviewed,
            "placement_stability": placement_stability,
            "family_support_level": family_support_level,
            "respite_available": respite_available,
            "education_engagement": education_engagement,
        }
    )

    # ---------------- Latent risk score -> admission label ----------------
    # Coefficients encode the evidence-based drivers of avoidable admission.
    # Effect sizes are set so the drivers carry a strong, learnable signal
    # (as expected of clinically meaningful features) while retaining enough
    # stochasticity that the problem is not perfectly separable.
    logit = (
        -1.0
        + 0.85 * num_previous_crises
        + 1.05 * num_previous_admissions
        + 0.45 * medication_changes_last_year
        + 0.010 * days_since_last_contact
        - 0.11 * community_contacts_last_year
        + 1.5 * (placement_stability == "Unstable")
        + 0.7 * (placement_stability == "At risk")
        + 1.2 * (family_support_level == "Low")
        + 0.5 * (family_support_level == "Medium")
        + 0.7 * (care_plan_reviewed == "No")
        + 0.5 * (respite_available == "No")
        + 0.9 * (education_engagement == "Not engaged")
        + 0.7 * (diagnosis == "Autism + LD")
    )
    # add moderate noise so the problem is realistic but not separable
    logit = logit + rng.normal(0, 0.5, size=n)

    prob = _sigmoid(logit)

    # Calibrate an intercept shift so overall prevalence ~ ADMISSION_RATE
    shift = 0.0
    for _ in range(60):
        cur = _sigmoid(logit + shift).mean()
        shift += (ADMISSION_RATE - cur) * 4.0
    prob = _sigmoid(logit + shift)

    df[TARGET] = rng.binomial(1, prob)

    if save:
        df.to_csv(DATASET_PATH, index=False)

    return df


if __name__ == "__main__":
    data = generate()
    print(f"Generated {len(data)} records -> {DATASET_PATH}")
    print(f"Admission rate: {data['admitted'].mean():.3f} "
          f"(target {ADMISSION_RATE})")
    print(data.head())
