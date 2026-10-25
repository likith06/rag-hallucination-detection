"""
01_data_preparation.py
Phase 1: Parse RAGTruth, create binary labels, stratified split.
Outputs: results/data_splits.pkl, results/data_stats.csv
"""

import os, json, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset", "RAGTruth-main", "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "Code", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RESPONSE_FILE    = os.path.join(DATASET_DIR, "response.jsonl")
SOURCE_INFO_FILE = os.path.join(DATASET_DIR, "source_info.jsonl")

print("=" * 60)
print("Phase 1 — Data Preparation")
print("=" * 60)

# ── Load response.jsonl ───────────────────────────────────────
print("\n[1] Loading response.jsonl...")
records = []
with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
df = pd.DataFrame(records)
print(f"    Total records loaded: {len(df):,}")
print(f"    Columns: {list(df.columns)}")

# ── Load source_info.jsonl ────────────────────────────────────
print("\n[2] Loading source_info.jsonl...")
sources = []
with open(SOURCE_INFO_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            sources.append(json.loads(line))
src_df = pd.DataFrame(sources)
print(f"    Total source records: {len(src_df):,}")

# ── Create binary hallucination label ────────────────────────
print("\n[3] Creating binary hallucination labels...")
def has_hallucination(labels):
    if isinstance(labels, list):
        return 1 if len(labels) > 0 else 0
    return 0

df["hallucinated"] = df["labels"].apply(has_hallucination)
df["num_spans"]    = df["labels"].apply(lambda x: len(x) if isinstance(x, list) else 0)

print(f"    Hallucinated:     {df['hallucinated'].sum():,} ({df['hallucinated'].mean()*100:.1f}%)")
print(f"    Clean responses:  {(1-df['hallucinated']).sum():,} ({(1-df['hallucinated']).mean()*100:.1f}%)")

# ── Merge with source info to get task_type ───────────────────
print("\n[4] Merging with source info...")
if "source_id" in df.columns and "source_id" in src_df.columns:
    df = df.merge(src_df[["source_id", "task_type"]], on="source_id", how="left")
    print(f"    Task type distribution:\n{df['task_type'].value_counts().to_string()}")
else:
    df["task_type"] = "unknown"

# ── Filter to QA subset ───────────────────────────────────────
print("\n[5] Filtering to QA subset...")
qa_df = df[df["task_type"] == "QA"].copy().reset_index(drop=True)
if len(qa_df) == 0:
    print("    WARNING: No QA records found — using full dataset")
    qa_df = df.copy().reset_index(drop=True)
else:
    print(f"    QA subset size: {len(qa_df):,}")
    print(f"    QA hallucinated: {qa_df['hallucinated'].sum():,} ({qa_df['hallucinated'].mean()*100:.1f}%)")

# ── Stratified 70/15/15 split ────────────────────────────────
print("\n[6] Creating 70/15/15 stratified split...")
train_df, temp_df = train_test_split(
    qa_df, test_size=0.30, stratify=qa_df["hallucinated"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["hallucinated"], random_state=42
)

print(f"    Train: {len(train_df):,}  |  Hall: {train_df['hallucinated'].sum():,} ({train_df['hallucinated'].mean()*100:.1f}%)")
print(f"    Val:   {len(val_df):,}   |  Hall: {val_df['hallucinated'].sum():,} ({val_df['hallucinated'].mean()*100:.1f}%)")
print(f"    Test:  {len(test_df):,}   |  Hall: {test_df['hallucinated'].sum():,} ({test_df['hallucinated'].mean()*100:.1f}%)")

# ── Save splits ───────────────────────────────────────────────
splits = {"train": train_df, "val": val_df, "test": test_df, "qa_full": qa_df, "full": df}
with open(os.path.join(RESULTS_DIR, "data_splits.pkl"), "wb") as f:
    pickle.dump(splits, f)
print(f"\n[7] Saved data_splits.pkl")

# ── Save stats CSV ────────────────────────────────────────────
stats = pd.DataFrame({
    "split":       ["train", "val", "test"],
    "total":       [len(train_df), len(val_df), len(test_df)],
    "hallucinated":[train_df["hallucinated"].sum(), val_df["hallucinated"].sum(), test_df["hallucinated"].sum()],
    "clean":       [(1-train_df["hallucinated"]).sum(), (1-val_df["hallucinated"]).sum(), (1-test_df["hallucinated"]).sum()],
    "hall_pct":    [train_df["hallucinated"].mean()*100, val_df["hallucinated"].mean()*100, test_df["hallucinated"].mean()*100],
})
stats.to_csv(os.path.join(RESULTS_DIR, "data_stats.csv"), index=False)
print("    Saved data_stats.csv")

# ── Visualisations ────────────────────────────────────────────
print("\n[8] Generating visualisations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Phase 1 — Data Preparation Overview", fontsize=14, fontweight="bold", y=1.01)

# Plot 1: Class distribution in QA subset
ax = axes[0, 0]
counts = qa_df["hallucinated"].value_counts().sort_index()
bars = ax.bar(["Clean (0)", "Hallucinated (1)"], counts.values,
              color=["#2a78d6", "#e34948"], edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{val:,}\n({val/len(qa_df)*100:.1f}%)",
            ha="center", fontsize=10, fontweight="bold")
ax.set_title("Class Distribution — QA Subset", fontweight="bold")
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.2)
ax.spines[["top", "right"]].set_visible(False)

# Plot 2: Task type distribution (full dataset)
ax = axes[0, 1]
task_counts = df["task_type"].value_counts()
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"][:len(task_counts)]
bars = ax.barh(task_counts.index, task_counts.values,
               color=colors, edgecolor="white")
for bar, val in zip(bars, task_counts.values):
    ax.text(val + 20, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=10, fontweight="bold")
ax.set_title("Task Type Distribution — Full Dataset", fontweight="bold")
ax.set_xlabel("Count")
ax.spines[["top", "right"]].set_visible(False)

# Plot 3: Split sizes
ax = axes[1, 0]
split_labels = ["Train (70%)", "Validation (15%)", "Test (15%)"]
split_sizes  = [len(train_df), len(val_df), len(test_df)]
split_hall   = [train_df["hallucinated"].sum(), val_df["hallucinated"].sum(), test_df["hallucinated"].sum()]
split_clean  = [s - h for s, h in zip(split_sizes, split_hall)]
x = np.arange(len(split_labels))
w = 0.35
b1 = ax.bar(x - w/2, split_clean, w, label="Clean", color="#2a78d6", edgecolor="white")
b2 = ax.bar(x + w/2, split_hall,  w, label="Hallucinated", color="#e34948", edgecolor="white")
ax.set_title("Train / Val / Test Split Sizes", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(split_labels)
ax.set_ylabel("Count")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)

# Plot 4: Hallucination span count distribution
ax = axes[1, 1]
span_counts = qa_df[qa_df["num_spans"] > 0]["num_spans"]
ax.hist(span_counts, bins=range(1, span_counts.max()+2), color="#eb6834",
        edgecolor="white", linewidth=0.8, align="left")
ax.set_title("Hallucination Span Count per Response\n(hallucinated responses only)", fontweight="bold")
ax.set_xlabel("Number of hallucination spans")
ax.set_ylabel("Count")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase1_data_overview.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("    Saved phase1_data_overview.png")
print("\n✓ Phase 1 complete. Next: python 02_rq1_classifier.py")
