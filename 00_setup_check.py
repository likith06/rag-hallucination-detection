"""
00_setup_check.py
Verifies all dependencies and dataset files are present before running the pipeline.
Run this first: python 00_setup_check.py
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

print("=" * 60)
print("RAGTruth Hallucination Detection — Setup Check")
print("=" * 60)

# ── 1. Python version ─────────────────────────────────────────
print(f"\n[1] Python version: {sys.version.split()[0]}", end="")
if sys.version_info >= (3, 8):
    print(" ✓")
else:
    print(" ✗  Need Python 3.8+")

# ── 2. Required packages ──────────────────────────────────────
packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "sentence_transformers": "sentence-transformers",
    "spacy": "spacy",
    "shap": "shap",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "torch": "torch",
}

print("\n[2] Package check:")
all_ok = True
for module, pip_name in packages.items():
    try:
        __import__(module)
        print(f"    {pip_name:<30} ✓")
    except ImportError:
        print(f"    {pip_name:<30} ✗  run: pip install {pip_name}")
        all_ok = False

# ── 3. spaCy model ────────────────────────────────────────────
print("\n[3] spaCy model:")
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("    en_core_web_sm                 ✓")
except OSError:
    print("    en_core_web_sm                 ✗  run: python -m spacy download en_core_web_sm")
    all_ok = False

# ── 4. Dataset files ──────────────────────────────────────────
print("\n[4] Dataset files:")
required_files = {
    "RAGTruth response.jsonl":
        os.path.join(DATASET_DIR, "RAGTruth-main", "dataset", "response.jsonl"),
    "RAGTruth source_info.jsonl":
        os.path.join(DATASET_DIR, "RAGTruth-main", "dataset", "source_info.jsonl"),
}

for name, path in required_files.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"    {name:<35} ✓  ({size:.1f} MB)")
    else:
        print(f"    {name:<35} ✗  Not found at: {path}")
        all_ok = False

# RAGTruth_Xtended (optional but needed for RQ2)
rtx_path = os.path.join(DATASET_DIR, "RAGTruth_Xtended", "rtx")
print(f"\n[5] RAGTruth_Xtended logits (needed for RQ2):")
if os.path.exists(rtx_path):
    models = os.listdir(rtx_path)
    for m in models:
        print(f"    {m:<40} ✓")
else:
    print(f"    rtx/ folder not found.")
    print(f"    Run restore.sh inside Dataset/RAGTruth_Xtended/ first.")

# ── 5. Results directory ──────────────────────────────────────
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)
print(f"\n[6] Results directory: {results_dir}  ✓")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_ok:
    print("All checks passed. Ready to run the pipeline.")
    print("Next: python 01_data_preparation.py")
else:
    print("Some checks failed. Fix the issues above before proceeding.")
print("=" * 60)
