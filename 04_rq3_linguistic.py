"""
04_rq3_linguistic.py
Phase 4 — RQ3: Linguistic feature extraction, correlation analysis, Shapley values.
Features: hedge density, NER density, type-token ratio, mean token length, response length.
Outputs: results/rq3_*.png, results/rq3_features.csv, results/rq3_results.csv
"""

import os, re, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "Code", "results")

print("=" * 60)
print("Phase 4 — RQ3: Linguistic Feature Analysis")
print("=" * 60)

# ── Load splits ───────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "data_splits.pkl"), "rb") as f:
    splits = pickle.load(f)
train_df = splits["train"]
val_df   = splits["val"]
test_df  = splits["test"]

# ── Hedge word lexicon ────────────────────────────────────────
HEDGE_WORDS = {
    "about", "almost", "apparently", "approximately", "around", "assume",
    "assumed", "assumption", "basically", "believed", "certain", "claim",
    "claimed", "claims", "could", "doubt", "essentially", "estimate",
    "estimated", "feel", "felt", "generally", "guess", "historically",
    "if", "imply", "indicate", "indicated", "indicates", "largely",
    "likely", "mainly", "may", "maybe", "might", "mostly", "nearly",
    "often", "perhaps", "possibly", "presumably", "probably", "quite",
    "rather", "reportedly", "roughly", "seem", "seemed", "seems",
    "should", "sometimes", "somewhat", "suggest", "suggested", "suggests",
    "suppose", "supposedly", "typically", "unclear", "uncertain",
    "understood", "unlikely", "usually", "vaguely", "would",
}

# ── Feature extraction ────────────────────────────────────────
print("\n[1] Loading spaCy model...")
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    USE_SPACY = True
    print("    spaCy en_core_web_sm loaded ✓")
except Exception as e:
    USE_SPACY = False
    print(f"    spaCy not available: {e}")
    print("    Using simple tokenisation fallback.")

def extract_features(text):
    text = str(text)
    tokens = text.lower().split()
    if not tokens:
        return {
            "hedge_density": 0.0, "ner_density": 0.0, "type_token_ratio": 0.0,
            "mean_token_length": 0.0, "response_length": 0, "sent_count": 0,
            "avg_sent_length": 0.0, "punct_density": 0.0,
        }
    n = len(tokens)

    # Hedge density
    hedge_count = sum(1 for t in tokens if t in HEDGE_WORDS)
    hedge_density = hedge_count / n

    # Type-token ratio (lexical diversity)
    ttr = len(set(tokens)) / n

    # Mean token length
    mean_tok_len = np.mean([len(t) for t in tokens])

    # Sentence count and avg sentence length
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = len(sentences)
    avg_sent_len = n / max(sent_count, 1)

    # Punctuation density
    punct_count = sum(1 for c in text if c in ".,;:!?-()[]\"'")
    punct_density = punct_count / max(len(text), 1)

    # NER density (named entities / tokens)
    if USE_SPACY:
        doc = nlp(text[:1000])  # cap for speed
        ner_count = len(doc.ents)
        ner_density = ner_count / n
    else:
        # Simple fallback: count capitalised words not at sentence start
        caps = sum(1 for t in tokens[1:] if t and t[0].isupper())
        ner_density = caps / n

    return {
        "hedge_density":    round(hedge_density, 6),
        "ner_density":      round(ner_density, 6),
        "type_token_ratio": round(ttr, 6),
        "mean_token_length":round(mean_tok_len, 6),
        "response_length":  n,
        "sent_count":       sent_count,
        "avg_sent_length":  round(avg_sent_len, 6),
        "punct_density":    round(punct_density, 6),
    }

FEATURE_COLS = [
    "hedge_density", "ner_density", "type_token_ratio",
    "mean_token_length", "response_length", "sent_count",
    "avg_sent_length", "punct_density"
]

print("\n[2] Extracting linguistic features from all splits...")
def extract_df(df, name):
    rows = []
    for i, row in df.iterrows():
        feats = extract_features(row["response"])
        feats["hallucinated"] = row["hallucinated"]
        feats["id"] = row.get("id", i)
        rows.append(feats)
    out = pd.DataFrame(rows)
    print(f"    {name}: {len(out)} rows extracted")
    return out

train_feat = extract_df(train_df, "train")
val_feat   = extract_df(val_df,   "val")
test_feat  = extract_df(test_df,  "test")

# Save features
test_feat.to_csv(os.path.join(RESULTS_DIR, "rq3_features.csv"), index=False)
print("    Saved rq3_features.csv")

# ── Point-biserial correlations ───────────────────────────────
print("\n[3] Point-biserial correlations with hallucination label...")
all_feat = pd.concat([train_feat, val_feat, test_feat], ignore_index=True)
corr_results = []
for feat in FEATURE_COLS:
    r, p = stats.pointbiserialr(all_feat["hallucinated"], all_feat[feat])
    corr_results.append({"Feature": feat, "r": round(r, 4), "p-value": round(p, 6),
                          "significant": p < 0.05})
    print(f"    {feat:<22} r={r:+.4f}  p={p:.4f}  {'*' if p<0.05 else ''}")
corr_df = pd.DataFrame(corr_results).sort_values("r", key=abs, ascending=False)

# ── Logistic regression with linguistic features ──────────────
print("\n[4] Training logistic regression on linguistic features...")
X_train = train_feat[FEATURE_COLS].values
X_val   = val_feat[FEATURE_COLS].values
X_test  = test_feat[FEATURE_COLS].values
y_train = train_feat["hallucinated"].values
y_val   = val_feat["hallucinated"].values
y_test  = test_feat["hallucinated"].values

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

ling_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
ling_lr.fit(X_train_sc, y_train)

y_pred = ling_lr.predict(X_test_sc)
y_prob = ling_lr.predict_proba(X_test_sc)[:, 1]
auc    = roc_auc_score(y_test, y_prob)
f1     = f1_score(y_test, y_pred)
print(f"    Linguistic LR — AUC-ROC={auc:.4f}  F1={f1:.4f}")

# ── SHAP feature importance ───────────────────────────────────
print("\n[5] Computing SHAP values...")
try:
    import shap
    explainer  = shap.LinearExplainer(ling_lr, X_train_sc, feature_names=FEATURE_COLS)
    shap_vals  = explainer(X_test_sc)
    mean_shap  = np.abs(shap_vals.values).mean(axis=0)
    shap_df    = pd.DataFrame({"Feature": FEATURE_COLS, "Mean |SHAP|": mean_shap})
    shap_df    = shap_df.sort_values("Mean |SHAP|", ascending=False)
    USE_SHAP   = True
    print("    SHAP values computed ✓")
    print(shap_df.to_string(index=False))
except Exception as e:
    USE_SHAP = False
    print(f"    SHAP not available: {e}  — using LR coefficients instead.")
    coefs   = np.abs(ling_lr.coef_[0])
    shap_df = pd.DataFrame({"Feature": FEATURE_COLS, "Mean |SHAP|": coefs})
    shap_df = shap_df.sort_values("Mean |SHAP|", ascending=False)

shap_df.to_csv(os.path.join(RESULTS_DIR, "rq3_shap_importance.csv"), index=False)
corr_df.to_csv(os.path.join(RESULTS_DIR, "rq3_correlations.csv"), index=False)
print("    Saved rq3_shap_importance.csv and rq3_correlations.csv")

# ── Visualisations ────────────────────────────────────────────
print("\n[6] Generating RQ3 visualisations...")
fig = plt.figure(figsize=(18, 14))
fig.suptitle("RQ3 — Linguistic Feature Analysis", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

feature_labels = {
    "hedge_density": "Hedge Density",
    "ner_density": "NER Density",
    "type_token_ratio": "Type-Token Ratio",
    "mean_token_length": "Mean Token Length",
    "response_length": "Response Length",
    "sent_count": "Sentence Count",
    "avg_sent_length": "Avg Sentence Length",
    "punct_density": "Punctuation Density",
}

# Plot 1: Correlation bar chart (colour-coded by direction)
ax1 = fig.add_subplot(gs[0, :2])
corr_sorted = corr_df.copy()
colors_c    = ["#e34948" if r > 0 else "#2a78d6" for r in corr_sorted["r"]]
bars = ax1.barh([feature_labels.get(f, f) for f in corr_sorted["Feature"]],
                corr_sorted["r"], color=colors_c, edgecolor="white", height=0.6)
ax1.axvline(0, color="black", linewidth=0.8)
for bar, val, sig in zip(bars, corr_sorted["r"], corr_sorted["significant"]):
    x_pos = val + 0.003 if val >= 0 else val - 0.003
    ha    = "left" if val >= 0 else "right"
    ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
             f"{val:+.4f}{'*' if sig else ''}",
             va="center", ha=ha, fontsize=9, fontweight="bold")
ax1.set_title("Point-Biserial Correlations with Hallucination Label\n(* = p < 0.05; red = positive, blue = negative correlation)",
              fontweight="bold")
ax1.set_xlabel("Pearson r")
ax1.spines[["top", "right"]].set_visible(False)

# Plot 2: SHAP importance
ax2 = fig.add_subplot(gs[0, 2])
shap_plot = shap_df.head(8)
colors_s   = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(shap_plot)))
bars2 = ax2.barh([feature_labels.get(f, f) for f in shap_plot["Feature"]],
                 shap_plot["Mean |SHAP|"], color=colors_s, edgecolor="white")
for bar, val in zip(bars2, shap_plot["Mean |SHAP|"]):
    ax2.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=9)
ax2.set_title("Feature Importance\n(Mean |SHAP| value)", fontweight="bold")
ax2.set_xlabel("Mean |SHAP|")
ax2.invert_yaxis()
ax2.spines[["top", "right"]].set_visible(False)

# Plots 3-5: Violin plots for top 3 features by |r|
top3 = corr_df.head(3)["Feature"].tolist()
for idx, feat in enumerate(top3):
    ax = fig.add_subplot(gs[1, idx])
    data_viol = [
        all_feat[all_feat["hallucinated"]==0][feat].values,
        all_feat[all_feat["hallucinated"]==1][feat].values,
    ]
    vp = ax.violinplot(data_viol, positions=[0, 1], showmedians=True, showmeans=False)
    for i, (body, color) in enumerate(zip(vp["bodies"], ["#2a78d6", "#e34948"])):
        body.set_facecolor(color); body.set_alpha(0.7)
    vp["cmedians"].set_color("black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Clean", "Hallucinated"])
    ax.set_title(f"{feature_labels.get(feat, feat)}\n(r={corr_df[corr_df['Feature']==feat]['r'].values[0]:+.4f})",
                 fontweight="bold", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

# Plot 6: Feature correlation heatmap
ax6 = fig.add_subplot(gs[2, :2])
feat_corr_matrix = all_feat[FEATURE_COLS].corr()
mask = np.triu(np.ones_like(feat_corr_matrix, dtype=bool))
sns.heatmap(feat_corr_matrix,
            xticklabels=[feature_labels.get(f, f) for f in FEATURE_COLS],
            yticklabels=[feature_labels.get(f, f) for f in FEATURE_COLS],
            annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=ax6, linewidths=0.3, cbar_kws={"shrink": 0.7},
            mask=mask)
ax6.set_title("Feature Intercorrelation Heatmap", fontweight="bold")
ax6.tick_params(axis="x", rotation=45, labelsize=8)
ax6.tick_params(axis="y", rotation=0,  labelsize=8)

# Plot 7: Summary table
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis("off")
td = [[feature_labels.get(r["Feature"], r["Feature"]),
       f"{r['r']:+.4f}", f"{r['p-value']:.4f}", "Yes" if r["significant"] else "No"]
      for _, r in corr_df.iterrows()]
tbl = ax7.table(cellText=td,
                colLabels=["Feature", "r", "p-value", "Sig."],
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.9)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#4a3aa7"); cell.set_text_props(color="white", fontweight="bold")
    elif td[row-1][3] == "Yes" if row > 0 else False:
        cell.set_facecolor("#f0fff0")
    cell.set_edgecolor("#cccccc")
ax7.set_title("Correlation Summary", fontweight="bold", pad=12)

plt.savefig(os.path.join(RESULTS_DIR, "rq3_linguistic_results.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("    Saved rq3_linguistic_results.png")
print("\n✓ Phase 4 complete. Next: python 05_ensemble.py")
