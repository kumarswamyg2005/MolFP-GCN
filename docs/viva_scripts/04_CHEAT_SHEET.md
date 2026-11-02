# ⚡ Quick Revision Cheat Sheet (All Members)

Print this or keep it on your phone. Glance at it before walking into the viva.

---

## 🔤 The One-Liner
"GCN learns a task-specific molecular fingerprint from atom graphs; beats ECFP+RF on ESOL
and Tox21 with scaffold splits."

---

## 📊 Results Table (Memorise)

| | ESOL RMSE ↓ | ESOL R² ↑ | Tox21 AUC ↑ |
|---|---|---|---|
| **GCN** | **1.063 ± 0.163** | **0.709** | **0.811 ± 0.009** |
| RF | 1.694 ± 0.498 | 0.267 | 0.756 ± 0.015 |

GCN wins 11/12 Tox21 assays. RF wins only SR-ARE.

---

## 🧬 Pipeline in 5 Seconds
SMILES → RDKit → 35-dim atom features → 3 GCNConv layers → mean+max pool → **256-d
fingerprint** → MLP → prediction

---

## 📐 The Equation
**H' = D^{−½}(A+I)D^{−½} H W**
- A+I = self-loop (keep own identity)
- D^{−½} = degree normalisation (fair averaging)
- W = shared learned weights

---

## 🛡️ Anti-Leakage Guarantees (Impress the Panel)
1. Scaffold split — no scaffold crosses sets
2. Test set used once — scramble test → bit-identical weights
3. Same split for both models — hash-checked
4. Train-only normalisation stats
5. All numbers auto-generated from results JSON

---

## ⚠️ Caveats to Mention FIRST (Shows Maturity)
- Seed 1 ESOL inflates the gap (PCBs in test)
- Tox21 AUC above expected band (easier split than DeepChem)
- Hyperparameters not tuned (fair but maybe not optimal)
- 3 seeds = rough error bar, not significance test
- GCNConv ignores bond types (C=O ≡ C–O)

---

## 🔮 Future Work (Top 3)
1. Edge-aware layer (GINE/GATv2) to use bond features
2. DeepChem deterministic split for comparable Tox21 numbers
3. Saliency validation by atom deletion

---

## 🎯 Who Covers What

| Topic | Who |
|-------|-----|
| GCN architecture, equation, residuals, pooling | **Kumaraswamy** |
| Featurisation (35-dim atoms, graph rules) | **Kumaraswamy** |
| Saliency / explainability | **Kumaraswamy** |
| Data loading, scaffold splitting algorithm | **Simon Peter** |
| Training loop, loss functions, early stopping | **Simon Peter** |
| Evaluation, metrics, reproducibility tests | **Simon Peter** |
| ECFP4 + RF baseline comparison | **Rakesh** |
| Ablation experiments (depth, readout, split) | **Rakesh** |
| Demo app, limitations, future work | **Rakesh** |
| **Shared overview, results, project motivation** | **Everyone** |

---

## 🔢 Numbers to Have Ready

| Fact | Number |
|------|--------|
| ESOL molecules | 1,128 |
| Tox21 molecules | 7,823 |
| Atom feature dimension | 35 |
| Hidden dimension | 128 |
| Fingerprint dimension | 256 |
| GCN layers | 3 |
| ECFP bits | 2,048 |
| RF trees | 500 |
| Tox21 assays | 12 |
| Tox21 missing labels | 17% |
| Tox21 positive rate | 7.5% |
| Train/val/test | 80/10/10 |
| Seeds | 0, 1, 2 |
| Patience | 30 epochs |
| Max epochs | 300 |
| Learning rate | 1e-3 |
| Dropout | 0.2 |
| Total tests | 34 |
| `make all` time | 11 min (M2 CPU) |

---

Good luck! 🍀
