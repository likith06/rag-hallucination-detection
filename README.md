# Inference-Time Hallucination Detection in RAG
## 7005SCN Individual Research Project — Likith Mallavarapu (16934674)

---

## Project Structure
```
Project/
├── Dataset/                    ← place RAGTruth-main and RAGTruth_Xtended here
│   ├── RAGTruth-main/
│   │   └── dataset/
│   │       ├── response.jsonl
│   │       └── source_info.jsonl
│   └── RAGTruth_Xtended/
│       └── rtx/                ← after running restore.sh
├── Code/
│   ├── 00_setup_check.py       ← verify environment and dataset
│   ├── 01_data_preparation.py  ← Phase 1: parse, label, split
│   ├── 02_rq1_classifier.py    ← Phase 2: embedding classifier
│   ├── 03_rq2_uncertainty.py   ← Phase 3: uncertainty quantification
│   ├── 04_rq3_linguistic.py    ← Phase 4: linguistic feature analysis
│   ├── 05_ensemble.py          ← Phase 5: ensemble + final evaluation
│   ├── 06_visualisations.py    ← all publication-quality plots
│   └── results/                ← all outputs saved here
└── README.md
```

## Setup

```bash
pip install pandas numpy scikit-learn sentence-transformers spacy shap matplotlib seaborn torch
python -m spacy download en_core_web_sm
```

## Run Order
```bash
cd Code
python 00_setup_check.py        # check everything is in place
python 01_data_preparation.py   # creates data splits
python 02_rq1_classifier.py     # RQ1 results
python 03_rq2_uncertainty.py    # RQ2 results
python 04_rq3_linguistic.py     # RQ3 results
python 05_ensemble.py           # final ensemble
python 06_visualisations.py     # all figures
```

All outputs (CSVs, PNGs, model files) are saved to `Code/results/`.

## Research Questions
- **RQ1**: Can a lightweight binary classifier on sentence embeddings detect hallucinations?
- **RQ2**: How do softmax entropy, min-token probability, and perplexity compare?
- **RQ3**: Which linguistic features correlate most strongly with hallucination labels?

## Dataset
- Primary: RAGTruth (Niu et al., ACL 2024)
- Logits: RAGTruth_Xtended (Snel, 2024, Hugging Face)
- Stretch: HaluEval (Li et al., EMNLP 2023)
