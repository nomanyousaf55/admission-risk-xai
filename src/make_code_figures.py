"""Render key code excerpts as syntax-highlighted images for the dissertation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from config import OUTPUTS_DIR

# simple python syntax colouring
KEYWORDS = {"def", "return", "for", "in", "if", "else", "elif", "import",
            "from", "class", "with", "as", "not", "and", "or", "lambda",
            "True", "False", "None", "try", "except"}
C_BG = "#1e1e2e"
C_TEXT = "#cdd6f4"
C_KW = "#cba6f7"
C_STR = "#a6e3a1"
C_COM = "#6c7086"
C_NUM = "#fab387"
C_FN = "#89b4fa"


def colour_tokens(line):
    """Yield (text, colour) tuples for one line."""
    if line.strip().startswith("#"):
        return [(line, C_COM)]
    out, buf, i = [], "", 0
    while i < len(line):
        ch = line[i]
        if ch in "\"'":
            if buf:
                out.append((buf, C_TEXT)); buf = ""
            q = ch; j = i + 1
            while j < len(line) and line[j] != q:
                j += 1
            out.append((line[i:j + 1], C_STR))
            i = j + 1
            continue
        if ch == "#":
            if buf:
                out.append((buf, C_TEXT)); buf = ""
            out.append((line[i:], C_COM))
            return out
        if ch.isalnum() or ch == "_":
            buf += ch
        else:
            if buf:
                col = C_KW if buf in KEYWORDS else (
                    C_NUM if buf.isdigit() else C_TEXT)
                out.append((buf, col)); buf = ""
            out.append((ch, C_TEXT))
        i += 1
    if buf:
        out.append((buf, C_KW if buf in KEYWORDS else C_TEXT))
    return out


def render(code, title, filename, width=11.5):
    lines = code.strip("\n").split("\n")
    h = 0.30 * len(lines) + 1.1
    fig, ax = plt.subplots(figsize=(width, h), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, len(lines) + 2.2)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.4, 0.3), 99.2, len(lines) + 1.5,
                                boxstyle="round,pad=0.2",
                                fc=C_BG, ec="#45475a", lw=1.2))
    ax.text(2, len(lines) + 1.25, title, color="#f9e2af",
            fontsize=11, fontweight="bold", family="monospace")
    for n, line in enumerate(lines):
        y = len(lines) - n
        ax.text(2.2, y, f"{n+1:2}", color=C_COM, fontsize=8.5,
                family="monospace", va="center")
        x = 5.2
        for text, col in colour_tokens(line):
            ax.text(x, y, text, color=col, fontsize=9,
                    family="monospace", va="center")
            x += len(text) * 0.72
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved", filename)


# ---------------- Figure: SMOTE pipeline ----------------
render('''
def build_pipeline(estimator):
    # SMOTE sits INSIDE the pipeline, so it is fitted only on
    # the training folds - no synthetic data reaches the test set
    return ImbPipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("smote",        SMOTE(random_state=RANDOM_STATE)),
        ("clf",          estimator),
    ])

search = GridSearchCV(build_pipeline(estimator), grid,
                      scoring="average_precision",
                      cv=StratifiedKFold(n_splits=5, shuffle=True),
                      n_jobs=-1)
search.fit(X_train, y_train)
''', "model_training.py - imbalance-aware pipeline and tuning",
   "code_pipeline.png")

# ---------------- Figure: evaluation metrics ----------------
render('''
def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return {
        "ROC-AUC":   roc_auc_score(y_test, proba),
        "PR-AUC":    average_precision_score(y_test, proba),
        "Recall":    recall_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "F1":        f1_score(y_test, pred),
    }
''', "model_training.py - imbalance-appropriate evaluation metrics",
   "code_metrics.png")

# ---------------- Figure: SHAP ----------------
render('''
def build_explainer(model, X_background):
    clf = model.named_steps["clf"]
    Xt_bg, names = _transform(model, X_background)
    tree_types = ("RandomForest", "GradientBoosting", "XGB")
    if any(t in type(clf).__name__ for t in tree_types):
        explainer = shap.TreeExplainer(clf)          # exact, fast
    else:
        bg = shap.sample(Xt_bg, min(100, len(Xt_bg)), random_state=0)
        explainer = shap.Explainer(clf.predict_proba, bg)
    return explainer, names

# one-hot columns are aggregated back to the original feature
# so each driver reads as a single unmet-need signal
for name, val in zip(names, sv):
    base = name
    for cat in CATEGORICAL_FEATURES:
        if name.startswith(cat + "_"):
            base = cat
    agg[base] = agg.get(base, 0.0) + float(val)
''', "explainability.py - SHAP explainer and unmet-need aggregation",
   "code_shap.png")

# ---------------- Figure: recommendation engine ----------------
render('''
RECOMMENDATION_RULES = [
    {"driver_label": "Recent crisis / behavioural escalation",
     "action":  "Activate crisis / intensive-support team early",
     "trigger": lambda r: r.get("num_previous_crises", 0) >= 2},
    {"driver_label": "Long gap since last service contact",
     "action":  "Schedule proactive review / care-coordination visit",
     "trigger": lambda r: r.get("days_since_last_contact", 0) >= 120},
]

def recommend(individual, shap_local=None):
    shap_map = dict(zip(shap_local["feature"], shap_local["shap_value"]))
    recs = [r for r in RECOMMENDATION_RULES if r["trigger"](individual)]
    # rank by SHAP impact so the biggest driver comes first
    recs.sort(key=lambda r: shap_map.get(r["driver_feature"], 0), reverse=True)
    return recs
''', "recommendation_engine.py - SHAP-driven care actions",
   "code_recommend.png")

print("All code figures written to", OUTPUTS_DIR)
