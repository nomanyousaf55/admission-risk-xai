"""
Community-Care Recommendation Engine (Chapter 3.4 — Table 3.1).

A transparent, rule-based module that converts the risk drivers surfaced by
SHAP into preventative community-care actions. It is deliberately rule-based
(not learned) so that it is deterministic, auditable and editable by
clinicians — in keeping with the transparency requirements for high-stakes
settings.

A recommendation is raised when a rule's trigger condition is met for the
individual; the SHAP local explanation is used to order recommendations so
the highest-impact drivers appear first.
"""
from config import RECOMMENDATION_RULES


def recommend(individual: dict, shap_local=None):
    """
    individual : dict of raw feature values for one person.
    shap_local : optional DataFrame from explainability.local_explanation
                 (columns: feature, shap_value) used to rank recommendations.
    Returns a list of dicts: driver, action, (shap_impact).
    """
    shap_map = {}
    if shap_local is not None:
        shap_map = dict(zip(shap_local["feature"], shap_local["shap_value"]))

    recs = []
    for rule in RECOMMENDATION_RULES:
        try:
            active = rule["trigger"](individual)
        except Exception:
            active = False
        if active:
            recs.append(
                {
                    "driver": rule["driver_label"],
                    "action": rule["action"],
                    "feature": rule["driver_feature"],
                    "shap_impact": float(shap_map.get(rule["driver_feature"], 0.0)),
                }
            )

    # rank by SHAP impact (positive = pushes risk up) when available
    recs.sort(key=lambda r: r["shap_impact"], reverse=True)
    return recs


if __name__ == "__main__":
    example = {
        "num_previous_crises": 3,
        "community_contacts_last_year": 2,
        "placement_stability": "Unstable",
        "family_support_level": "Low",
        "days_since_last_contact": 200,
    }
    for r in recommend(example):
        print(f"- {r['driver']}  ->  {r['action']}")
