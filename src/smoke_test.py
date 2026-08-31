"""Quick smoke test of SHAP + recommendation engine end to end."""
import numpy as np
import joblib

from config import BEST_MODEL_PATH
from preprocessing import load_data, split_X_y
from explainability import build_explainer, global_importance, local_explanation
from recommendation_engine import recommend

model = joblib.load(BEST_MODEL_PATH)
df = load_data()
X, y = split_X_y(df)
expl, names = build_explainer(model, X.sample(200, random_state=1))

print("=== GLOBAL feature importance (top 8) ===")
gi = global_importance(expl, model, X.sample(300, random_state=2))
print(gi[["label", "importance"]].head(8).to_string(index=False))

proba = model.predict_proba(X)[:, 1]
idx = int(np.argmax(proba))
row = X.iloc[[idx]]
print(f"\n=== LOCAL explanation for idx={idx} (risk={proba[idx]:.2f}) ===")
le = local_explanation(expl, model, row)
print(le[["label", "shap_value"]].head(6).to_string(index=False))

print("\n=== RECOMMENDATIONS ===")
recs = recommend(row.iloc[0].to_dict(), le)
for r in recs:
    print(f"- {r['driver']}  ->  {r['action']}")
print("\nSMOKE TEST OK")
