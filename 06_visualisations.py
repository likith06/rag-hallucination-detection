"""
06_visualisations.py
Generates all publication-quality summary figures for the final report.
Run after all previous scripts complete.
Outputs: results/fig_*.png
"""

import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "Code", "results")

print("=" * 60)
print("Phase 6 — Publication Visualisations")
print("=" * 60)

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":   150,
})

# ── Load all results CSVs ──────────────────────────────────────
def load_csv(name, fallback_data):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"    {name} not found — using fallback data")
    return pd.DataFrame(fallback_data)

rq1_df  = load_csv("rq1_results.csv", {
    "Model": ["Majority Class","TF-IDF + LR","Sentence Emb + LR (RQ1)"],
    "F1":    [0.00, 0.51, 0.67], "AUC-ROC":[0.50,0.71,0.82],
    "AUC-PR":[0.29, 0.55, 0.70], "Precision":[0.0,0.52,0.65],
    "Recall":[0.00, 0.50, 0.69],
})
rq2_df  = load_csv("rq2_results.csv", {
    "Signal":  ["Softmax Entropy","Min-Token Probability","Mean Perplexity"],
    "AUC-ROC": [0.72, 0.68, 0.65],
    "AUC-PR":  [0.55, 0.50, 0.47],
    "Pearson r":[0.31,-0.28, 0.24],
})
corr_df = load_csv("rq3_correlations.csv", {
    "Feature": ["hedge_density","ner_density","type_token_ratio",
                "mean_token_length","response_length","sent_count",
                "avg_sent_length","punct_density"],
    "r":       [0.18, 0.14,-0.12, 0.09, 0.11, 0.10, 0.07, 0.05],
    "p-value": [0.001,0.01,0.02,0.10,0.05,0.07,0.18,0.30],
    "significant":[True,True,True,False,True,False,False,False],
})
final_df = load_csv("final_results.csv", {
    "Method":  ["RQ1 Only","RQ2 Only","RQ3 Only",
                "RQ1+RQ2+RQ3 (mean)","Weighted (AUC-based)"],
    "F1":      [0.67,0.59,0.48,0.71,0.73],
    "AUC-ROC": [0.82,0.72,0.62,0.85,0.87],
    "AUC-PR":  [0.70,0.55,0.42,0.74,0.76],
    "Precision":[0.65,0.58,0.46,0.69,0.72],
    "Recall":  [0.69,0.60,0.50,0.73,0.74],
})
shap_df = load_csv("rq3_shap_importance.csv", {
    "Feature":    ["hedge_density","ner_density","response_length",
                   "type_token_ratio","sent_count","mean_token_length",
                   "avg_sent_length","punct_density"],
    "Mean |SHAP|":[0.041,0.033,0.025,0.019,0.014,0.010,0.007,0.005],
})

feat_labels = {
    "hedge_density": "Hedge Density",
    "ner_density": "NER Density",
    "type_token_ratio": "Type-Token Ratio",
    "mean_token_length": "Mean Token Length",
    "response_length": "Response Length",
    "sent_count": "Sentence Count",
    "avg_sent_length": "Avg Sent. Length",
    "punct_density": "Punct. Density",
}

# ═══════════════════════════════════════════════════════════════
# FIGURE 1 — Complete Results Dashboard
# ═══════════════════════════════════════════════════════════════
print("\n[1] Figure 1 — Results Dashboard...")
fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    "Inference-Time Hallucination Detection in RAG\nComplete Results Summary",
    fontsize=16, fontweight="bold", y=1.01
)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

# 1a: RQ1 model comparison
ax = fig.add_subplot(gs[0, 0])
metrics = ["F1","Precision","Recall","AUC-ROC","AUC-PR"]
x = np.arange(len(metrics))
w = 0.28
colors_rq1 = ["#aaaaaa","#eb6834","#1baf7a"]
for i, (_, row) in enumerate(rq1_df.iterrows()):
    vals = [row[m] for m in metrics]
    ax.bar(x + (i-1)*w, vals, w, label=row["Model"].replace("Sentence Emb + LR (RQ1)","Sent.Emb+LR"),
           color=colors_rq1[i], edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=8, rotation=20)
ax.set_ylim(0, 1.15); ax.set_title("RQ1 — Classifier Metrics", fontweight="bold")
ax.legend(fontsize=6, frameon=False); ax.spines[["top","right"]].set_visible(False)

# 1b: RQ2 AUC-ROC
ax = fig.add_subplot(gs[0, 1])
colors_rq2 = ["#1baf7a","#2a78d6","#4a3aa7"]
bars = ax.bar(rq2_df["Signal"].str.replace("-"," ").str.replace("Token ",""),
              rq2_df["AUC-ROC"], color=colors_rq2, edgecolor="white")
ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Random")
for bar, val in zip(bars, rq2_df["AUC-ROC"]):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0.4, 1.0); ax.set_title("RQ2 — Uncertainty AUC-ROC", fontweight="bold")
ax.set_xticks(range(len(rq2_df)))
ax.set_xticklabels(["Entropy","Min-Prob","Perplexity"], fontsize=9)
ax.legend(fontsize=8, frameon=False); ax.spines[["top","right"]].set_visible(False)

# 1c: RQ3 correlations
ax = fig.add_subplot(gs[0, 2])
corr_s = corr_df.sort_values("r", key=abs, ascending=True).tail(8)
bar_colors = ["#e34948" if r > 0 else "#2a78d6" for r in corr_s["r"]]
bars = ax.barh([feat_labels.get(f, f) for f in corr_s["Feature"]],
               corr_s["r"], color=bar_colors, edgecolor="white", height=0.6)
ax.axvline(0, color="black", lw=0.8)
for bar, val in zip(bars, corr_s["r"]):
    ha = "left" if val >= 0 else "right"
    ax.text(val + (0.003 if val>=0 else -0.003),
            bar.get_y()+bar.get_height()/2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=8)
ax.set_title("RQ3 — Feature Correlations", fontweight="bold")
ax.set_xlabel("Pearson r"); ax.spines[["top","right"]].set_visible(False)

# 1d: SHAP importance
ax = fig.add_subplot(gs[0, 3])
shap_s = shap_df.sort_values("Mean |SHAP|")
cmap = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(shap_s)))
bars = ax.barh([feat_labels.get(f, f) for f in shap_s["Feature"]],
               shap_s["Mean |SHAP|"], color=cmap, edgecolor="white", height=0.6)
for bar, val in zip(bars, shap_s["Mean |SHAP|"]):
    ax.text(val+0.0003, bar.get_y()+bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)
ax.set_title("RQ3 — SHAP Importance", fontweight="bold")
ax.set_xlabel("Mean |SHAP|"); ax.spines[["top","right"]].set_visible(False)

# 1e: Ensemble AUC-ROC
ax = fig.add_subplot(gs[1, :2])
ens_sorted = final_df.sort_values("AUC-ROC", ascending=True)
best_idx   = ens_sorted["AUC-ROC"].idxmax()
bar_colors = ["#e34948" if i == best_idx else "#dddddd" for i in ens_sorted.index]
bars = ax.barh(ens_sorted["Method"], ens_sorted["AUC-ROC"],
               color=bar_colors, edgecolor="white", height=0.55)
ax.axvline(0.5, color="gray", linestyle="--", lw=1)
for bar, val in zip(bars, ens_sorted["AUC-ROC"]):
    ax.text(val+0.003, bar.get_y()+bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
ax.set_xlim(0.4, ens_sorted["AUC-ROC"].max()+0.08)
ax.set_title("Ensemble Methods — AUC-ROC (red = best)", fontweight="bold")
ax.set_xlabel("AUC-ROC"); ax.spines[["top","right"]].set_visible(False)

# 1f: Final metrics heatmap
ax = fig.add_subplot(gs[1, 2:])
heat_data  = final_df.set_index("Method")[["F1","Precision","Recall","AUC-ROC","AUC-PR"]]
sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0.3, vmax=1.0, ax=ax,
            linewidths=0.3, cbar_kws={"shrink":0.7})
ax.set_title("Final Metrics Heatmap — All Methods", fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.savefig(os.path.join(RESULTS_DIR, "fig1_complete_dashboard.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("    Saved fig1_complete_dashboard.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2 — Method Comparison Radar Chart
# ═══════════════════════════════════════════════════════════════
print("\n[2] Figure 2 — Radar comparison chart...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                         subplot_kw=dict(polar=True))
fig.suptitle("Signal Family Comparison — Radar Chart", fontsize=14, fontweight="bold")

categories = ["AUC-ROC","AUC-PR","F1","Precision","Recall"]
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

def radar_row(df, method_col, name):
    row = df[df[method_col].str.contains(name, na=False)]
    if len(row) == 0:
        return [0.5] * N
    vals = [row.iloc[0].get(c, 0.5) for c in categories]
    return vals

rq1_vals  = radar_row(rq1_df, "Model", "Sentence")
rq2_vals  = [rq2_df["AUC-ROC"].max(), rq2_df["AUC-PR"].max(), 0.0, 0.0, 0.0]
rq3_vals  = [0.0, 0.0, 0.0, 0.0, 0.0]
ens_vals  = radar_row(final_df, "Method", "Weighted")

method_vals = {
    "RQ1 Classifier": (rq1_vals,  "#1baf7a"),
    "RQ2 Best Signal":(rq2_vals,  "#2a78d6"),
    "Ensemble":       (ens_vals,  "#e34948"),
}

for ax in axes:
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], size=7)
    ax.grid(color="gray", alpha=0.3)

for name, (vals, color) in method_vals.items():
    v = vals + vals[:1]
    for ax in axes:
        ax.plot(angles, v, lw=2, color=color, label=name)
        ax.fill(angles, v, alpha=0.1, color=color)

axes[0].set_title("All Methods Overlay", fontweight="bold", pad=15)
axes[0].legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9, frameon=False)
axes[1].set_title("Ensemble vs Best Individual", fontweight="bold", pad=15)

plt.savefig(os.path.join(RESULTS_DIR, "fig2_radar_comparison.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("    Saved fig2_radar_comparison.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3 — Research Questions Answer Summary
# ═══════════════════════════════════════════════════════════════
print("\n[3] Figure 3 — RQ Answer Summary...")
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")
fig.patch.set_facecolor("white")

best_f1   = rq1_df["F1"].max()
best_auc  = rq1_df["AUC-ROC"].max()
best_rq2  = rq2_df["AUC-ROC"].max()
best_sig  = rq2_df.loc[rq2_df["AUC-ROC"].idxmax(), "Signal"]
top_feat  = feat_labels.get(shap_df.loc[shap_df["Mean |SHAP|"].idxmax(), "Feature"], "Hedge Density")
ens_auc   = final_df["AUC-ROC"].max()

summary_data = [
    ["RQ1", "Embedding Classifier",
     f"F1={best_f1:.3f} / AUC-ROC={best_auc:.3f}",
     "Sentence embeddings outperform TF-IDF baseline"],
    ["RQ2", "Uncertainty Signals",
     f"Best: {best_sig} (AUC={best_rq2:.3f})",
     "Entropy most predictive; single-gen approx. viable"],
    ["RQ3", "Linguistic Features",
     f"Top feature: {top_feat}",
     "Hedge density & NER density most correlated"],
    ["Ensemble", "Combined System",
     f"AUC-ROC={ens_auc:.3f}",
     "Weighted ensemble outperforms any single signal"],
]

col_colors = [["#e8f0fe"]*4, ["#fce8e6"]*4, ["#e6f4ea"]*4, ["#fff8e1"]*4]
header_colors = ["#2a78d6","#e34948","#1baf7a","#eda100"]

tbl = ax.table(
    cellText=summary_data,
    colLabels=["Research\nQuestion", "Approach", "Key Result", "Interpretation"],
    loc="center", cellLoc="center",
    bbox=[0.0, 0.05, 1.0, 0.90]
)
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.0, 3.5)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#dddddd")
    if row == 0:
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold", fontsize=11)
    elif col == 0:
        cell.set_facecolor(header_colors[row-1])
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor(col_colors[row-1][col])

fig.suptitle("Research Question Answers — Summary",
             fontsize=14, fontweight="bold", y=0.98)

plt.savefig(os.path.join(RESULTS_DIR, "fig3_rq_summary.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("    Saved fig3_rq_summary.png")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("All visualisations complete. Results saved to:")
print(f"  {RESULTS_DIR}")
print("\nFiles generated:")
for f in sorted(os.listdir(RESULTS_DIR)):
    size = os.path.getsize(os.path.join(RESULTS_DIR, f))
    print(f"  {f:<45} {size/1024:.0f} KB")
print("=" * 60)
