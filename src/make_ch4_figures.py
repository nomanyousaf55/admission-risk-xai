"""Generate publication-quality figures for Chapter 4 (Experimental Results)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from config import METADATA_PATH, RESULTS_PATH, OUTPUTS_DIR
from preprocessing import load_data, split_X_y
from explainability import build_explainer, global_importance

meta = joblib.load(METADATA_PATH)
results = pd.read_csv(RESULTS_PATH)
FIGDIR = OUTPUTS_DIR
plt.rcParams.update({"font.family": "serif", "font.size": 11})

# ---- Fig 4.1 : model comparison bar chart ----
fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
metrics = ["ROC-AUC", "PR-AUC", "Recall", "F1"]
x = np.arange(len(results))
w = 0.2
colors = ["#2980b9", "#c0392b", "#27ae60", "#8e44ad"]
for i, m in enumerate(metrics):
    ax.bar(x + i * w, results[m], w, label=m, color=colors[i])
ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels(results["Model"], rotation=20, ha="right")
ax.set_ylabel("Score")
ax.set_title("Figure 4.1  Comparative performance of the five models")
ax.legend(ncol=4, fontsize=9)
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR / "fig41_model_comparison.png", bbox_inches="tight")
plt.close()

# ---- Fig 4.2 : ROC + PR curves for best model ----
roc = meta["roc_curve"]; pr = meta["pr_curve"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
a1.plot(roc["fpr"], roc["tpr"], color="#2980b9", lw=2,
        label=f"{meta['best_name']} (AUC={results.iloc[0]['ROC-AUC']:.3f})")
a1.plot([0, 1], [0, 1], "--", color="grey")
a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate")
a1.set_title("(a) ROC curve"); a1.legend(loc="lower right"); a1.grid(alpha=0.3)
a2.plot(pr["recall"], pr["precision"], color="#c0392b", lw=2,
        label=f"PR-AUC={results.iloc[0]['PR-AUC']:.3f}")
a2.axhline(meta["test_admission_rate"], ls="--", color="grey",
           label=f"Baseline={meta['test_admission_rate']:.3f}")
a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
a2.set_title("(b) Precision-Recall curve"); a2.legend(loc="upper right"); a2.grid(alpha=0.3)
fig.suptitle("Figure 4.2  ROC and Precision-Recall curves (best model)")
plt.tight_layout()
plt.savefig(FIGDIR / "fig42_curves.png", bbox_inches="tight")
plt.close()

# ---- Fig 4.3 : confusion matrix ----
cm = np.array(meta["confusion_matrix"])
fig, ax = plt.subplots(figsize=(5, 4.2), dpi=200)
im = ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center",
            color="white" if v > cm.max() / 2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: No", "Pred: Yes"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: No", "Actual: Yes"])
ax.set_title("Figure 4.3  Confusion matrix (best model)")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(FIGDIR / "fig43_confusion.png", bbox_inches="tight")
plt.close()

# ---- Fig 4.4 : global SHAP importance ----
model = joblib.load(__import__("config").BEST_MODEL_PATH)
df = load_data(); X, y = split_X_y(df)
expl = build_explainer(model, X.sample(150, random_state=1))[0]
gi = global_importance(expl, model, X.sample(250, random_state=2)).head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
ax.barh(gi["label"], gi["importance"], color="#16a085")
ax.set_xlabel("Mean |SHAP| value")
ax.set_title("Figure 4.4  Global feature importance (SHAP)")
plt.tight_layout()
plt.savefig(FIGDIR / "fig44_shap_global.png", bbox_inches="tight")
plt.close()

print("Saved figures 4.1-4.4 to", FIGDIR)
print(gi[["label", "importance"]].to_string(index=False))
