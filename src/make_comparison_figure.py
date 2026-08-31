"""Figure comparing this study's results with published readmission studies."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUTS_DIR

plt.rcParams.update({"font.family": "serif", "font.size": 11})

studies = [
    ("Kansagara et al. (2011)\nreview of 26 models", 0.60, "#95a5a6", "Traditional indices"),
    ("Frizzell et al. (2017)\nn=56,477", 0.62, "#95a5a6", "ML vs regression"),
    ("Futoma et al. (2015)\nlarge admin data", 0.68, "#95a5a6", "Penalised LR / RF / NN"),
    ("Rajkomar et al. (2018)\nn=216,221, full EHR", 0.755, "#7f8c8d", "Deep learning"),
    ("This study — real data\nUCI, n=101,763", 0.622, "#c0392b", "LR + SMOTE + SHAP"),
    ("This study — synthetic\nautism/LD, n=3,000", 0.873, "#27ae60", "LR + SMOTE + SHAP"),
]

fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=200)
y = np.arange(len(studies))
vals = [s[1] for s in studies]
cols = [s[2] for s in studies]
bars = ax.barh(y, vals, color=cols, height=0.62)

ax.set_yticks(y)
ax.set_yticklabels([s[0] for s in studies], fontsize=9.5)
ax.invert_yaxis()
ax.set_xlim(0, 1.12)
ax.set_xlabel("ROC-AUC / C-statistic")
ax.axvline(0.5, ls="--", color="#bdc3c7", lw=1)
ax.text(0.505, len(studies) - 0.35, "chance", fontsize=8, color="#7f8c8d")
ax.grid(axis="x", alpha=0.3)

for b, s in zip(bars, studies):
    ax.text(b.get_width() + 0.012, b.get_y() + b.get_height() / 2,
            f"{s[1]:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.text(0.012, b.get_y() + b.get_height() / 2, s[3],
            va="center", fontsize=8, color="white", style="italic")

ax.set_title("Figure 4.7  This study compared with published "
             "admission/readmission prediction studies",
             fontsize=11.5, pad=12)

# legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="#95a5a6", label="Published — traditional / classical ML"),
    Patch(facecolor="#7f8c8d", label="Published — deep learning on full EHR"),
    Patch(facecolor="#c0392b", label="This study — real data"),
    Patch(facecolor="#27ae60", label="This study — synthetic (target population)"),
], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
   fontsize=8.5, framealpha=0.95)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "fig47_literature_comparison.png", bbox_inches="tight")
plt.close()
print("saved fig47_literature_comparison.png")
