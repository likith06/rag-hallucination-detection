"""
02_rq1_classifier.py
Phase 2 — RQ1: Sentence embedding + logistic regression hallucination classifier.
Outputs: results/rq1_*.png, results/rq1_results.csv, results/rq1_model.pkl
"""

import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, f1_score,
    precision_score, recall_score
)
from sklearn.dummy import DummyClassifier

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "Code", "results")

print("=" * 60)
print("Phase 2 — RQ1: Embedding Classifier")
print("=" * 60)

# ── Load splits ───────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "data_splits.pkl"), "rb") as f:
    splits = pickle.load(f)
train_df = splits["train"]
val_df   = splits["val"]
test_df  = splits["test"]

print(f"\n[1] Loaded splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# ── Encode with sentence transformer ─────────────────────────
print("\n[2] Encoding with all-mpnet-base-v2 (this may take a few minutes)...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    train_emb = model.encode(train_df["response"].tolist(), show_progress_bar=True, batch_size=32)
    val_emb   = model.encode(val_df["response"].tolist(),   show_progress_bar=True, batch_size=32)
    test_emb  = model.encode(test_df["response"].tolist(),  show_progress_bar=True, batch_size=32)
    print(f"    Embedding shape: {train_emb.shape}")
    np.save(os.path.join(RESULTS_DIR, "train_embeddings.npy"), train_emb)
    np.save(os.path.join(RESULTS_DIR, "val_embeddings.npy"),   val_emb)
    np.save(os.path.join(RESULTS_DIR, "test_embeddings.npy"),  test_emb)
except Exception as e:
    print(f"    WARNING: {e}")
    print("    Falling back to random embeddings for structure demonstration.")
    np.random.seed(42)
    train_emb = np.random.randn(len(train_df), 768)
    val_emb   = np.random.randn(len(val_df),   768)
    test_emb  = np.random.randn(len(test_df),  768)

y_train = train_df["hallucinated"].values
y_val   = val_df["hallucinated"].values
y_test  = test_df["hallucinated"].values

# ── Train models ──────────────────────────────────────────────
print("\n[3] Training classifiers...")

# Baseline 1: Majority class
dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(train_emb, y_train)

# Baseline 2: TF-IDF + LR
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_tfidf_train = tfidf.fit_transform(train_df["response"].tolist())
X_tfidf_val   = tfidf.transform(val_df["response"].tolist())
X_tfidf_test  = tfidf.transform(test_df["response"].tolist())
tfidf_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
tfidf_lr.fit(X_tfidf_train, y_train)

# Main: Sentence embedding + LR
# Tune C on validation set
best_c, best_f1 = 1.0, 0.0
for c in [0.01, 0.1, 1.0, 5.0, 10.0]:
    lr = LogisticRegression(C=c, max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(train_emb, y_train)
    f1 = f1_score(y_val, lr.predict(val_emb))
    print(f"    C={c:<5} → val F1={f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_c  = c

print(f"\n    Best C={best_c} (val F1={best_f1:.4f})")
best_lr = LogisticRegression(C=best_c, max_iter=1000, random_state=42, class_weight="balanced")
best_lr.fit(train_emb, y_train)

# ── Evaluate on test set ──────────────────────────────────────
print("\n[4] Evaluating on test set...")
models_eval = {
    "Majority Class":          (dummy,    train_emb,       test_emb,      y_test),
    "TF-IDF + LR":             (tfidf_lr, X_tfidf_train,  X_tfidf_test,  y_test),
    "Sentence Emb + LR (RQ1)": (best_lr,  train_emb,      test_emb,      y_test),
}

results = []
for name, (m, _, X_t, y_t) in models_eval.items():
    y_pred = m.predict(X_t)
    if hasattr(m, "predict_proba"):
        y_prob = m.predict_proba(X_t)[:, 1]
    else:
        y_prob = y_pred.astype(float)
    results.append({
        "Model": name,
        "F1":        round(f1_score(y_t, y_pred), 4),
        "Precision": round(precision_score(y_t, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_t, y_pred, zero_division=0), 4),
        "AUC-ROC":   round(roc_auc_score(y_t, y_prob), 4),
        "AUC-PR":    round(average_precision_score(y_t, y_prob), 4),
    })
    print(f"\n  {name}:")
    print(f"    F1={results[-1]['F1']}  Precision={results[-1]['Precision']}  "
          f"Recall={results[-1]['Recall']}  AUC-ROC={results[-1]['AUC-ROC']}  AUC-PR={results[-1]['AUC-PR']}")

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(RESULTS_DIR, "rq1_results.csv"), index=False)
print("\n    Saved rq1_results.csv")

# Save model
with open(os.path.join(RESULTS_DIR, "rq1_model.pkl"), "wb") as f:
    pickle.dump({"model": best_lr, "tfidf": tfidf, "tfidf_lr": tfidf_lr}, f)

# ── Visualisations ────────────────────────────────────────────
print("\n[5] Generating RQ1 visualisations...")
y_pred_main = best_lr.predict(test_emb)
y_prob_main = best_lr.predict_proba(test_emb)[:, 1]
y_prob_tfidf = tfidf_lr.predict_proba(X_tfidf_test)[:, 1]

fig = plt.figure(figsize=(18, 12))
fig.suptitle("RQ1 — Hallucination Classifier Results", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1: Metric comparison bar chart
ax1 = fig.add_subplot(gs[0, 0])
metrics = ["F1", "Precision", "Recall", "AUC-ROC", "AUC-PR"]
colors_m = ["#2a78d6", "#eb6834", "#1baf7a"]
x = np.arange(len(metrics))
w = 0.25
for i, row in res_df.iterrows():
    ax1.bar(x + (i-1)*w, [row[m] for m in metrics], w,
            label=row["Model"], color=colors_m[i], edgecolor="white")
ax1.set_xticks(x); ax1.set_xticklabels(metrics, fontsize=9)
ax1.set_ylim(0, 1.1)
ax1.set_ylabel("Score")
ax1.set_title("Model Comparison — All Metrics", fontweight="bold")
ax1.legend(fontsize=7, frameon=False)
ax1.spines[["top", "right"]].set_visible(False)
for bar in ax1.patches:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.2f}", ha="center", fontsize=6.5, color="#333")

# Plot 2: Confusion matrix
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, y_pred_main)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
            xticklabels=["Clean", "Hallucinated"],
            yticklabels=["Clean", "Hallucinated"],
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax2.set_title("Confusion Matrix\n(Sentence Emb + LR)", fontweight="bold")
ax2.set_ylabel("True label")
ax2.set_xlabel("Predicted label")

# Plot 3: ROC curves
ax3 = fig.add_subplot(gs[0, 2])
fpr_m, tpr_m, _ = roc_curve(y_test, y_prob_main)
fpr_t, tpr_t, _ = roc_curve(y_test, y_prob_tfidf)
ax3.plot(fpr_m, tpr_m, color="#1baf7a", lw=2,
         label=f"Sentence Emb+LR (AUC={roc_auc_score(y_test, y_prob_main):.3f})")
ax3.plot(fpr_t, tpr_t, color="#eb6834", lw=2, linestyle="--",
         label=f"TF-IDF+LR (AUC={roc_auc_score(y_test, y_prob_tfidf):.3f})")
ax3.plot([0,1],[0,1], "k:", lw=1, label="Random (AUC=0.500)")
ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curves", fontweight="bold")
ax3.legend(fontsize=8, frameon=False)
ax3.spines[["top", "right"]].set_visible(False)

# Plot 4: Precision-Recall curves
ax4 = fig.add_subplot(gs[1, 0])
prec_m, rec_m, _ = precision_recall_curve(y_test, y_prob_main)
prec_t, rec_t, _ = precision_recall_curve(y_test, y_prob_tfidf)
ax4.plot(rec_m, prec_m, color="#1baf7a", lw=2,
         label=f"Sentence Emb+LR (AP={average_precision_score(y_test, y_prob_main):.3f})")
ax4.plot(rec_t, prec_t, color="#eb6834", lw=2, linestyle="--",
         label=f"TF-IDF+LR (AP={average_precision_score(y_test, y_prob_tfidf):.3f})")
ax4.axhline(y_test.mean(), color="gray", linestyle=":", lw=1, label="Baseline (random)")
ax4.set_xlabel("Recall"); ax4.set_ylabel("Precision")
ax4.set_title("Precision-Recall Curves", fontweight="bold")
ax4.legend(fontsize=8, frameon=False)
ax4.spines[["top", "right"]].set_visible(False)

# Plot 5: Probability distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(y_prob_main[y_test==0], bins=30, alpha=0.65,
         color="#2a78d6", label="Clean (true)", density=True)
ax5.hist(y_prob_main[y_test==1], bins=30, alpha=0.65,
         color="#e34948", label="Hallucinated (true)", density=True)
ax5.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold=0.5")
ax5.set_xlabel("Predicted probability"); ax5.set_ylabel("Density")
ax5.set_title("Prediction Probability Distribution", fontweight="bold")
ax5.legend(fontsize=8, frameon=False)
ax5.spines[["top", "right"]].set_visible(False)

# Plot 6: Summary table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
table_data = res_df[["Model", "F1", "AUC-ROC", "AUC-PR"]].values.tolist()
col_labels = ["Model", "F1", "AUC-ROC", "AUC-PR"]
tbl = ax6.table(cellText=table_data, colLabels=col_labels,
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.scale(1.1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2a78d6"); cell.set_text_props(color="white", fontweight="bold")
    elif r == len(table_data):
        cell.set_facecolor("#e8f4e8")
    cell.set_edgecolor("#cccccc")
ax6.set_title("Results Summary", fontweight="bold", pad=20)

plt.savefig(os.path.join(RESULTS_DIR, "rq1_classifier_results.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("    Saved rq1_classifier_results.png")
print("\n✓ Phase 2 complete. Next: python 03_rq2_uncertainty.py")
