# Viva Script — Simon Peter (S20230010171)

## Your Focus Areas
1. **Data Pipeline & Scaffold Splitting**
2. **Training Loop & Optimisation**
3. **Evaluation, Metrics & Reproducibility**

> Read [00_OVERVIEW.md](00_OVERVIEW.md) first for the project-level context everyone shares.

---

# PART 1: DATA PIPELINE

## 1.1 Where the Data Comes From

**Source:** DeepChem mirror of MoleculeNet benchmark datasets.

**Download:**
```bash
python scripts/download_data.py
```
- Downloads `delaney-processed.csv` (ESOL) and `tox21.csv` to `data/raw/`.
- SHA-256 checksums are verified against `data/CHECKSUMS.txt` — the script **refuses to
  continue** if a file has changed. This ensures reproducibility.

## 1.2 Data Loading (`data.py`)

The loading pipeline:
```
data/raw/delaney-processed.csv  →  pd.read_csv()
                                       │
                              parse_smiles() on each row
                                       │
                              Drop invalid SMILES (log the count)
                                       │
                              Return: (smiles_list, y_array[n × n_tasks], task_names)
```

**Key details:**
- ESOL: 1,128 molecules, 1 target column ("measured log solubility in mols per litre").
  None are invalid, so no rows are dropped.
- Tox21: 7,831 rows, but **8 have SMILES that RDKit can't sanitise** → dropped with a
  logged count → **7,823 usable molecules**, 12 target columns.
- **NaN = missing label** in Tox21. 17% of labels are NaN.
- The function is `@cache`-decorated, so it's loaded once and shared.

## 1.3 Graph Construction (`load_graphs` in `data.py`)

```python
@cache
def load_graphs(name):
    smiles, y, _ = load_dataset(name)
    return [smiles_to_data(s, yi) for s, yi in zip(smiles, y)]
```

- Each valid molecule becomes a PyG `Data` object with atom features (x), edges (edge_index),
  bond features (edge_attr), target (y), SMILES string, and atom count.
- **Row i here = index i in the split files.** This is the contract between data loading and
  splitting.

---

# PART 2: SCAFFOLD SPLITTING — THE HEART OF HONEST EVALUATION

## 2.1 What Is a Scaffold?

A **Bemis–Murcko scaffold** is the core ring structure of a molecule plus the linkers between
rings, with all side chains stripped off.

**Example:**
- Aspirin, ibuprofen, and salicylic acid share a benzene ring scaffold.
- Polychlorinated biphenyls (PCBs) share a biphenyl scaffold.

## 2.2 How the Scaffold Split Works (Know This Algorithm)

```
Step 1: Group molecules by scaffold SMILES
        (acyclic molecules all share the empty scaffold "")

Step 2: Sort groups into "big" (> half the val/test size) and "small"
        Big groups go first (shuffled by seed)
        Small groups go next (shuffled by seed)

Step 3: Greedily assign groups:
        → train while it stays ≤ 80%
        → val while it stays ≤ 10%
        → test (the rest, ~10%)

Rule: A WHOLE GROUP always lands in ONE split.
      No scaffold ever appears in more than one split.
```

**Split ratio:** 80% train / 10% validation / 10% test

**The split is saved to:** `data/splits/{dataset}_seed{n}.json`
- Contains: `{"train": [...], "val": [...], "test": [...], "hash": "...", ...}`
- The **hash** is SHA-256 of the index lists. It's checked on every load — if someone edits
  the file, it raises an error.

## 2.3 Why Scaffold Split Matters — Be Ready With This Argument

> "Chemistry datasets contain families of near-identical molecules — same ring system,
> different side chain. A random split puts relatives on both sides, so the model gets tested
> on molecules it has essentially seen. The scaffold split keeps each family together. This
> is closer to the real use case: predicting properties of genuinely new chemistry."

**Ablation proof (your money quote):**
| Split | GCN RMSE | RF RMSE |
|-------|----------|---------|
| Scaffold | 1.063 ± 0.163 | 1.694 ± 0.498 |
| Random | 0.800 ± 0.023 | 1.256 ± 0.037 |

Both models look much better on random split. That improvement is **leakage**, not real
performance. The scaffold split is honest.

## 2.4 Both Models See the Same Split (Invariant 8)

- `split.get_split()` is called by **both** the GCN trainer and the RF baseline.
- The split file is the **single source of truth**.
- `evaluate.py` asserts that the split hashes match between GCN and RF runs. If they don't,
  it raises a `RuntimeError`.

---

# PART 3: THE TRAINING LOOP

## 3.1 Training Configuration (YAML → Dataclass)

```yaml
# configs/esol_gcn.yaml
dataset: esol
name: gcn
split: scaffold
hidden: 128
layers: 3
dropout: 0.2
readout: meanmax
lr: 0.001            # learning rate
weight_decay: 0.00001 # L2 regularisation
batch_size: 64
max_epochs: 300
patience: 30          # early stopping patience
```

**Both ESOL and Tox21 use identical hyperparameters.** They were fixed in advance and not
tuned for either model — this keeps the comparison fair.

## 3.2 The Training Loop Step by Step

```
for each epoch (up to 300):
    1. Shuffle training molecules into batches of 64
    2. For each batch:
       a. Forward pass: model(batch) → predictions
       b. Compute loss (MSE for ESOL, masked BCE for Tox21)
       c. Backward pass: compute gradients
       d. Optimizer step: update weights
    3. Score on validation set
    4. If val metric improved → save checkpoint (best.pt)
    5. If no improvement for 30 epochs → STOP (early stopping)
```

## 3.3 ESOL Loss — MSE on Normalised Targets

```python
def target_stats(y, train_idx):
    t = y[train_idx]
    return {"mean": float(t.mean()), "std": float(t.std())}

loss = MSE(prediction, (y - train_mean) / train_std)
```

**Critical:** Normalisation stats come from the **train split only** (invariant 6). At
inference, predictions are de-normalised: `pred * std + mean`.

## 3.4 Tox21 Loss — Masked Binary Cross-Entropy

```python
def masked_bce(logits, y, pos_weight=None):
    m = ~torch.isnan(y)  # mask: True where labelled
    return BCE_with_logits(logits[m], y[m], pos_weight=pw[m])
```

**Two complications handled:**
1. **Missing labels (17%):** NaN values are **masked out**, never filled with 0.
   Filling with 0 would teach the model that untested molecules are non-toxic — that's false.
2. **Class imbalance (7.5% positive):** `pos_weight` = negatives/positives per assay,
   computed on the training set only. This upweights the rare toxic examples so the model
   can't score well by always predicting "not toxic".

## 3.5 Optimizer & Early Stopping

- **Adam** optimizer: lr=1e-3, weight_decay=1e-5 (L2 regularisation).
- **Early stopping:** patience=30 epochs. If the validation metric (RMSE for ESOL, mean
  ROC-AUC for Tox21) hasn't improved for 30 epochs, training stops and the best checkpoint
  is kept.
- Best epochs in practice: ESOL seeds 0,1,2 → epochs 24, 43, 25.
  Tox21 seeds 0,1,2 → epochs 73, 28, 56.

## 3.6 Seeding — Exact Reproducibility

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
```

The seed controls:
- The scaffold split (which groups go where)
- Model weight initialisation
- Batch shuffling order
- All random operations in NumPy/PyTorch

**Result:** `make all` from a fresh clone reproduced every committed result **byte for byte**
on an Apple M2 CPU in 11 minutes.

---

# PART 4: EVALUATION & METRICS

## 4.1 Evaluation Flow (`evaluate.py`)

```
For each dataset (esol / tox21):
    For each model (gcn / rf):
        For each seed (0, 1, 2):
            1. Load checkpoint (GCN) or train RF
            2. Predict on test split (ONCE only)
            3. Compute metrics
    Assert GCN and RF used the same split (hash check)
    Compute mean ± std across seeds
    Write results/{dataset}.json
```

## 4.2 Understanding the Metrics

### RMSE (Root Mean Squared Error) — ESOL
```python
rmse = sqrt(mean((y_true - y_pred)²))
```
- In logS units. RMSE of 1 means the prediction is typically off by 1 log unit (a factor of
  10 in solubility). Lower is better.

### MAE (Mean Absolute Error) — ESOL
```python
mae = mean(|y_true - y_pred|)
```
- Less sensitive to outliers than RMSE. Lower is better.

### R² (Coefficient of Determination) — ESOL
```python
r2 = 1 - SS_res / SS_tot
```
- 1 = perfect predictions, 0 = no better than guessing the mean. Higher is better.

### ROC-AUC (Area Under ROC Curve) — Tox21
- "If you pick one toxic and one non-toxic molecule at random, how often does the model rank
  the toxic one as more likely toxic?"
- 0.5 = coin flip, 1.0 = perfect ranking. Higher is better.
- Computed **per assay** on labelled rows only, then averaged across assays.
- Tasks where all test labels are one class get NaN and are excluded from the mean.

## 4.3 The Seed 1 ESOL Story — Know This Cold

**What happened:** In seed 1, the polychlorinated biphenyl (PCB) scaffold group all landed
in the test set. These are the **least soluble** molecules in ESOL (logS down to −11.6).

**Why the RF failed badly:**
- A random forest predicts by averaging training targets. It **can't predict below its
  training range** — it's bounded by the most extreme training example.
- ECFP bits record that chlorine is *present*, not *how many* there are. So all PCBs look
  similar in ECFP space.
- RF predicted −4.6 for the least soluble molecule; GCN predicted −7.2 (true: even lower).

**Per-seed results:**
| Seed | GCN RMSE | RF RMSE |
|------|----------|---------|
| 0 | 0.904 | 1.288 |
| 1 | 1.230 | **2.249** ← the PCB seed |
| 2 | 1.055 | 1.545 |

**Talking point:** "The GCN still wins on all three seeds, but the headline mean exaggerates
the gap because of seed 1. On a typical split (seeds 0 and 2), the margin is smaller."

## 4.4 The Tox21 ROC-AUC Caveat

Our Tox21 ROC-AUC of 0.811 is slightly above our expected band of 0.68–0.80.

**Explanation:** DeepChem's deterministic scaffold split sends the **rarest scaffolds** to
test, which is harder. Our split sends large groups to train first and samples test from
smaller groups at random — this is likely easier. So our numbers **should not be compared
directly with published papers**.

---

# PART 5: REPRODUCIBILITY GUARANTEES

## 5.1 The Integrity Tests You Should Mention

| Test | What It Proves |
|------|---------------|
| `test_scaffold_split_properties` | No scaffold crosses splits; ratios ≈ 80/10/10 |
| `test_train_never_sees_test` | Scramble test data, retrain → weights are bit-identical (test set unused during training) |
| `test_normalization_*` | Changing val/test targets doesn't change normalisation stats |
| `test_gcn_and_rf_use_same_split` | Both models read the same split file |
| `test_split_file_is_single_source_of_truth` | Both models follow even a deliberately altered split file |
| `test_permutation_invariance` | Reorder atoms → output changes by < 1e-5 |
| `test_masked_bce_ignores_nan` | NaN labels contribute zero to the loss |
| `test_masked_roc_auc_ignores_nan` | NaN labels excluded from ROC-AUC |

**Total:** 34 tests, all passing.

## 5.2 Checkpoint Contents

Each `best.pt` stores:
```python
{
    "state_dict": model weights,
    "config": full Config dataclass as dict,
    "seed": seed number,
    "out_dim": number of output dimensions,
    "norm": {"mean": ..., "std": ...} for ESOL, None for Tox21,
    "featurizer_version": "1",
    "split_hash": 16-char hex hash of split indices,
    "epoch": best epoch number,
    "val": validation metrics at best epoch,
}
```

If the featurizer version in the checkpoint doesn't match the current code, loading raises
an error. This prevents using a model trained with different features.

---

# POSSIBLE QUESTIONS FOR SIMON PETER

### Q1: "Why not tune hyperparameters?"
→ "We fixed all hyperparameters in advance to keep the comparison fair. Tuning one model
and not the other would be misleading. We also only have ~1,100 ESOL molecules — a
validation set of ~110 molecules is too small for reliable hyperparameter selection. A
tuned RF (or one with count-based fingerprints instead of bits) would probably close some
of the ESOL gap."

### Q2: "Why 3 seeds? Isn't that too few?"
→ "Three seeds give a rough error bar showing run-to-run spread, not a formal significance
test. With ESOL's 114 test molecules, one scaffold group can move the RMSE by more than the
difference between models (as seed 1 shows). For a proper significance test we'd need more
seeds and a paired test, but 3 seeds with full reporting of per-seed numbers is standard for
MoleculeNet papers."

### Q3: "How do you handle the class imbalance in Tox21?"
→ "Only 7.5% of labelled Tox21 entries are positive ('toxic'). Without correction, the model
could score well by always predicting 'not toxic'. We compute `pos_weight` = (number of
negatives / number of positives) per assay on the training set only, and pass it to BCE loss.
This upweights the rare positives. Side effect: predicted probabilities run high, so we tell
users to compare molecules rather than read probabilities as absolute risk."

### Q4: "What if someone edits the split file?"
→ "Every split file stores a SHA-256 hash of its index lists. `get_split()` recomputes the
hash on every load and raises a ValueError if it doesn't match. Additionally, `evaluate.py`
asserts that the GCN and RF split hashes match."

### Q5: "Why Adam and not SGD?"
→ "Adam adapts the learning rate per parameter, which helps on sparse molecular features
where some atoms/features appear rarely. We also use weight decay (1e-5) for L2
regularisation. We didn't compare optimizers — this is a standard choice."

### Q6: "What does early stopping actually save you from?"
→ "Without early stopping, the model would train for all 300 epochs and overfit the training
set. Early stopping monitors the validation metric and keeps the checkpoint from the epoch
where validation performance was best. With patience=30, it waits 30 epochs after the last
improvement before stopping, to avoid stopping too early on a noisy metric."

---

# QUICK REFERENCE CARD

| Item | Value |
|------|-------|
| Train/Val/Test ratio | 80/10/10 scaffold split |
| Split file location | `data/splits/{dataset}_seed{n}.json` |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-5 |
| Batch size | 64 |
| Max epochs | 300 |
| Early stopping patience | 30 epochs |
| ESOL loss | MSE on z-scored targets (train stats only) |
| Tox21 loss | Masked BCE with per-assay pos_weight |
| Seeds | 0, 1, 2 |
| ESOL test size | ~114 molecules (seed 0) |
| Tox21 test size | ~783 molecules (seed 0) |
| Total pytest tests | 34, all passing |
| Reproducibility | Byte-identical results from fresh clone |
| Key code files | `data.py`, `split.py`, `train.py`, `evaluate.py`, `metrics.py` |
