"""Figures for the real-data validation section (Chapter 4.5)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from config import OUTPUTS_DIR, MODELS_DIR

meta = joblib.load(MODELS_DIR / "metadata_real.joblib")
results = pd.DataFrame(meta["results"]).sort_values("PR-AUC", ascending=False)
plt.rcParams.update({"font.family": "serif", "font.size": 11})

# Fig 4.5 : model comparison on real data
fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
metrics = ["ROC-AUC", "PR-AUC", "Recall", "F1"]
x = np.arange(len(results)); w = 0.2
colors = ["#2980b9", "#c0392b", "#27ae60", "#8e44ad"]
for i, m in enumerate(metrics):
    ax.bar(x + i * w, results[m], w, label=m, color=colors[i])
ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels(results["Model"], rotation=20, ha="right")
ax.set_ylabel("Score"); ax.set_ylim(0, 1)
ax.set_title("Figure 4.5  Model performance on the real UCI dataset")
ax.legend(ncol=4, fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "fig45_real_comparison.png", bbox_inches="tight")
plt.close()

# Fig 4.6 : ROC + PR curves (real best model)
roc = meta["roc_curve"]; pr = meta["pr_curve"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
a1.plot(roc["fpr"], roc["tpr"], color="#2980b9", lw=2,
        label=f"{meta['best_name']} (AUC={results.iloc[0]['ROC-AUC']:.3f})")
a1.plot([0, 1], [0, 1], "--", color="grey")
a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate")
a1.set_title("(a) ROC curve — real data"); a1.legend(loc="lower right"); a1.grid(alpha=0.3)
a2.plot(pr["recall"], pr["precision"], color="#c0392b", lw=2,
        label=f"PR-AUC={results.iloc[0]['PR-AUC']:.3f}")
a2.axhline(meta["test_admission_rate"], ls="--", color="grey",
           label=f"Baseline={meta['test_admission_rate']:.3f}")
a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
a2.set_title("(b) Precision-Recall — real data"); a2.legend(loc="upper right"); a2.grid(alpha=0.3)
fig.suptitle("Figure 4.6  ROC and Precision-Recall curves on the real dataset")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "fig46_real_curves.png", bbox_inches="tight")
plt.close()

print("Saved real-data figures 4.5, 4.6")
