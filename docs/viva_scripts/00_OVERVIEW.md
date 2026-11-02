# Viva Script — Shared Overview (All Members Must Study This)

> **Read this first.** This section covers what every team member must know. Then read your
> own individual script for your assigned deep-dive topics.

---

## 🎯 One-Sentence Summary (Memorise This)

"We trained a graph convolutional network to learn a task-specific molecular fingerprint
from atom-level graphs, and showed it beats the traditional fixed ECFP fingerprint + random
forest on both ESOL solubility regression and Tox21 multi-label toxicity classification,
using honest scaffold splits and three seeded runs."

---

## 📌 Project Name & Motivation

**MolFP-GCN** — Learned Molecular Fingerprints with Graph Convolutional Networks.

**Why this project?**
- A molecule needs to be turned into numbers before any ML model can use it.
- The **old way** (ECFP / Morgan fingerprint): a fixed hash function looks at each atom's
  2-bond neighbourhood and writes bits. The same molecule always gets the same 2048 bits,
  regardless of whether you want to predict solubility or toxicity.
- **Our way** (GCN): treat the molecule as a graph (atoms = nodes, bonds = edges). A learned
  neural network aggregates neighbour information in rounds. The resulting 256-number vector
  is a **learned fingerprint** shaped by the training task.
- **Central question:** Does the learned fingerprint actually beat ECFP + random forest when
  both get the same data and the same honest split?

---

## 📊 Datasets (Know These Cold)

| Dataset | Task Type | Size | Target | Missing Labels |
|---------|-----------|------|--------|----------------|
| **ESOL** (Delaney 2004) | Regression | 1,128 molecules | logS (water solubility) | None |
| **Tox21** (MoleculeNet) | Multi-label classification | 7,823 molecules | 12 binary toxicity assays | 17% missing |

- **logS** is a log scale: −1 means 10× more soluble than −2, etc. Higher = dissolves better.
- **Tox21** has 12 assay tasks (NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD,
  NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53). Only 7.5% of known labels
  are positive ("toxic").
- Source: DeepChem mirror. SHA-256 checksums verified on download.
- RDKit couldn't sanitise 8 Tox21 SMILES → dropped. ESOL lost none.

---

## 🏆 Results (The Numbers Everyone Must Know)

| Model | ESOL RMSE ↓ | ESOL R² ↑ | Tox21 ROC-AUC ↑ |
|-------|-------------|-----------|------------------|
| **GCN (learned FP)** | **1.063 ± 0.163** | **0.709 ± 0.048** | **0.811 ± 0.009** |
| ECFP4 + Random Forest | 1.694 ± 0.498 | 0.267 ± 0.248 | 0.756 ± 0.015 |

**Key takeaways:**
1. GCN wins on both tasks.
2. On Tox21, the GCN is ahead on **11 of 12 assays** (RF ahead only on SR-ARE).
3. All numbers are mean ± std over **seeds 0, 1, 2** (scaffold split + model initialisation).

**Caveats to mention proactively:**
- The RF's ESOL average is dragged down by seed 1 (RMSE 2.25) where polychlorinated
  biphenyls all landed in the test set. On the other two seeds the gap is smaller.
- The Tox21 ROC-AUC 0.811 is slightly above our expected band (0.68–0.80). Our scaffold
  split is likely easier than DeepChem's deterministic one, so don't compare directly with papers.

---

## 🔑 What is a Scaffold Split? (Will Be Asked)

- A **scaffold** is the core ring structure of a molecule (Bemis–Murcko framework).
- We group molecules by scaffold and keep each **entire group** in one split (train / val / test).
- **Why?** Because chemistry datasets contain families of near-copies. If one family member
  is in training and its twin is in test, the test is too easy — the model has effectively
  "seen" the answer.
- **Ablation proof:** Random split gave GCN RMSE 0.80 ± 0.02 vs scaffold split
  1.06 ± 0.16. That improvement on random split is leakage, not real performance.

---

## 📐 Metrics Explained (So You Don't Freeze When Asked)

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **RMSE** | Typical size of error in logS units. Lower = better. | < 1.5 |
| **MAE** | Average absolute error. Lower = better. | < 1.0 |
| **R²** | Fraction of variance explained. 1 = perfect, 0 = just guessing the mean. | > 0.5 |
| **ROC-AUC** | If you pick one toxic and one non-toxic molecule at random, how often does the model rank correctly? 0.5 = coin flip, 1.0 = perfect. | > 0.75 |

---

## 🏗️ How To Run Everything (Demo Commands)

```bash
make all          # setup + download + train 3 seeds × 2 models × 2 datasets + evaluate + figures + tests
make demo         # launch Streamlit at http://localhost:8501
make ablations    # depth / readout / split ablation experiments
make test         # run 34 pytest tests
make lint         # ruff check + format
```

One single training run:
```bash
python -m molfp.train --config configs/esol_gcn.yaml --seed 0
```

---

## 📁 Project Layout (Point to folders confidently)

```
molfp/           → all ML code: featurize, model, train, evaluate, explain, figures
configs/         → YAML configs for each experiment
scripts/         → download_data.py
app/             → Streamlit demo (streamlit_app.py + style.css)
data/            → raw CSVs + scaffold splits
results/         → esol.json, tox21.json, ablations.json
reports/         → REPORT.md + figures/
tests/           → 34 pytest tests
docs/            → EXPLAIN.md, PROJECT_IN_SIMPLE_WORDS.md, ARCHITECTURE.md, DESIGN.md
```

---

## 🛡️ Key Integrity Guarantees (Mention These Proactively — They Impress)

1. **No leakage:** scaffold split tested with zero-overlap assertion on canonical SMILES.
2. **Test set used once:** a test scrambles test data and retrains — weights come out
   bit-identical, proving test set is never used during training.
3. **Both models see the same split:** split file is the single source of truth; evaluate
   asserts matching hashes between GCN and RF.
4. **Normalisation stats from train only:** changing val/test targets doesn't change them.
5. **Every number auto-generated:** `python -m molfp.figures` fills in all report numbers
   from `results/*.json`. Nothing is typed by hand.

---

> **Now read your individual script below.**
