# Viva Script — Rakesh (S20230010196)

## Your Focus Areas
1. **Baseline Comparison: ECFP4 + Random Forest**
2. **Ablation Experiments & Analysis**
3. **Demo App, Limitations & Future Work**

> Read [00_OVERVIEW.md](00_OVERVIEW.md) first for the project-level context everyone shares.

---

# PART 1: THE BASELINE — ECFP4 + RANDOM FOREST

## 1.1 What Is ECFP (Extended-Connectivity Fingerprint)?

**Opening line for the viva:**
> "ECFP is the standard way to turn a molecule into a fixed-length vector. It's a Morgan
> fingerprint with radius 2, producing 2048 bits. For each atom, a fixed hash function looks
> at the atom and its neighbours within 2 bonds, and sets one bit in a 2048-length vector.
> The same molecule always gets the same bits, regardless of the task."

**How ECFP4 works step by step:**
1. For each atom, collect its properties (element, bond count, charge, etc.)
2. Look at all atoms within **2 bonds** (that's the radius-2 part, hence "ECFP**4**" =
   diameter 4)
3. Hash the neighbourhood information into one of **2048 bit positions**
4. Set that bit to 1
5. The final fingerprint: a 2048-bit binary vector (mostly zeros)

**In our code (`baseline.py`):**
```python
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def ecfp(smiles):
    return np.stack([_MORGAN.GetFingerprintAsNumPy(parse_smiles(s)[0]) for s in smiles])
```

## 1.2 ECFP vs Learned Fingerprint — The Key Comparison

| Aspect | ECFP4 (Fixed) | GCN (Learned) |
|--------|--------------|---------------|
| **How it aggregates** | Fixed hash of atom neighbourhoods | Learned weighted average of atom features |
| **Neighbourhood radius** | 2 bonds | 3 bonds (3 GCN layers) |
| **Dimension** | 2048 bits | 256 floats |
| **Task-specific?** | No — same bits for any task | Yes — fingerprint shaped by training |
| **Bond types** | Included in the hash | Ignored by GCNConv |
| **Atom counts** | Presence only (bit is 1 or 0) | Captured via mean pooling + mass feature |
| **Differentiable?** | No — can't backprop through a hash | Yes — gradients flow end-to-end |
| **Gradient explanations?** | No | Yes (saliency maps) |

**Critical distinction (say this in the viva):**
> "The word 'differentiable' is the key. ECFP uses a fixed hash — you can't take its
> gradient, so you can't train it end-to-end. The GCN replaces the hash with smooth
> operations (averages, matrix products, ReLU), so the loss gradient flows all the way
> back through the fingerprint into the weights. The fingerprint learns to emphasise
> whatever matters for the task."

## 1.3 What Is a Random Forest?

A random forest is an ensemble of **500 decision trees**, each trained on a random subset
of the data and features. The final prediction is the average of all trees (regression) or
majority vote (classification).

**For ESOL:**
```python
rf = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
rf.fit(X[train], y[train, 0])
pred = rf.predict(X[test])
```

**For Tox21:**
```python
# ONE separate forest per assay, trained only on rows labelled for that assay
for t in range(12):
    labelled_train = train[~np.isnan(y[train, t])]
    rf = RandomForestClassifier(n_estimators=500, random_state=seed)
    rf.fit(X[labelled_train], y[labelled_train, t])
    pred[:, t] = rf.predict_proba(X[test])[:, 1]
```

**Why one forest per assay?** Because each assay has different molecules labelled. Training
one forest on all 12 tasks would need molecules labelled for all 12 — very few have that.

## 1.4 Why the RF Lost — Structural Reasons

1. **Can't extrapolate:** A random forest predicts by averaging training targets. It
   **cannot predict below its training range**. In seed 1, PCBs (logS down to −11.6) were
   all in test — the RF could only predict ≈ −4.6 (its training minimum).

2. **ECFP doesn't count atoms:** A bit is either 0 or 1. Three chlorines in a neighbourhood
   set the same bit as one chlorine. The GCN sees atom counts through mean pooling and the
   mass feature.

3. **No task-specific features:** ECFP gives the same bits whether predicting solubility or
   toxicity. The GCN tailors its representation to the loss.

4. **No gradient-based explanations:** ECFP + RF is a black box — you can compute feature
   importance but can't make atom-level saliency maps.

---

# PART 2: ABLATION EXPERIMENTS

## 2.1 What Is an Ablation?

> "An ablation study changes one component at a time to see if it matters. We ran three
> ablations on ESOL: depth (number of GCN layers), readout (pooling method), and split
> type (scaffold vs random)."

## 2.2 Depth Ablation — Number of GCN Layers

| Layers | RMSE |
|--------|------|
| 1 | 1.082 ± 0.204 |
| 2 | 1.002 ± 0.135 |
| **3 (default)** | **1.063 ± 0.163** |
| 4 | 1.053 ± 0.156 |

**Conclusion:** Every difference is within one standard deviation. We **cannot claim** 3
layers is best, or that depth matters at all at this dataset size.

**Why no over-smoothing at 4 layers?** Probably because of residual connections (skip
connections from layer 2 onwards), which let later layers add refinements rather than replace
the representation. ESOL is also too small to separate these settings.

**How to explain over-smoothing:**
> "With many layers, every atom ends up averaging over nearly the whole molecule. All atom
> vectors converge to the same value. Then max pooling can't pick out any distinctive
> features — everything looks the same."

## 2.3 Readout Ablation — Pooling Method

| Readout | RMSE |
|---------|------|
| Mean only | 0.985 ± 0.172 |
| Max only | 1.046 ± 0.072 |
| **Mean+Max (default)** | **1.063 ± 0.163** |

**Conclusion:** Again, no clear winner. All within noise. We kept mean+max because it was
the planned default, **not because the ablation favoured it**. (Say this — it shows honesty.)

**Why both mean and max?**
- **Mean** captures: "what is this molecule made of on average?"
- **Max** captures: "is there at least one atom of this kind?" (e.g., one nitro group
  can dominate toxicity but would be diluted by mean)

## 2.4 Split Ablation — Scaffold vs Random

| Split | GCN RMSE | RF RMSE |
|-------|----------|---------|
| **Scaffold** | **1.063 ± 0.163** | **1.694 ± 0.498** |
| Random | 0.800 ± 0.023 | 1.256 ± 0.037 |

**This is the one clear effect.** Both models improve dramatically on random split. This
improvement is **leakage** — the random split puts close analogues of test molecules in the
training set.

**Key observations:**
1. Both models improve, not just one — the leakage is universal.
2. The standard deviations shrink on random split (0.023 vs 0.163) — the split is more
   stable because there are no large scaffold groups to move around.
3. This is **why every headline number uses the scaffold split**.

## 2.5 How to Present Ablations in the Viva

> "We ran three ablations on ESOL. Depth and readout made no measurable difference — all
> variants were within noise. The split type was the one clear effect: switching from
> scaffold to random dropped the GCN's RMSE from 1.06 to 0.80, but that improvement is
> leakage from shared scaffolds between train and test, not real generalisation. We report
> this honestly to justify why all headline numbers use the scaffold split."

---

# PART 3: THE DEMO APP

## 3.1 What the Demo Does

A **Streamlit** web app at `http://localhost:8501`. The user can:

1. **Enter a SMILES string** or pick from 8 pre-loaded molecules (aspirin, caffeine,
   paracetamol, ibuprofen, benzene, ethanol, nicotine, and a known toxic molecule the model
   never saw in training).
2. Press **Predict**.
3. See:
   - The molecule drawn with atoms **coloured by saliency** (red = pushes prediction up,
     blue = pushes down)
   - Predicted **solubility** in logS with a plain-language band (soluble / moderately
     soluble / poorly soluble)
   - **12 toxicity probabilities** (with a note: "model probability", never "risk")
   - **3 closest training molecules** by cosine similarity of their 256-d learned
     fingerprints
4. Switch the colouring between solubility and any of the 12 toxicity assays.

**URL parameter:** `http://localhost:8501/?smiles=CCO` works directly.

## 3.2 Design Philosophy (From DESIGN.md)

| Element | Choice | Reason |
|---------|--------|--------|
| Palette | Paper white (#F6F8F7), ink black (#1C2321), one accent blue (#2F6FB0) | Lab-paper aesthetic, readable on projector |
| Font | Atkinson Hyperlegible (Google Fonts) | Designed for projector legibility |
| SMILES font | IBM Plex Mono | Code must show exact characters |
| Layout | Two columns desktop, stacked mobile | Left-aligned, clean |
| Molecule drawing | RDKit SVG, min 420px | Memorable element, must be readable from back of room |
| Saliency scale | Blue ↔ white ↔ red (diverging) | Data visualisation, not decoration |
| Motion | Saliency fade-in 250ms only | Respects prefers-reduced-motion |

**Forbidden:** Rounded cards with shadows, gradient headers, all-caps, emoji, rainbow
colormaps, Streamlit default red accent.

## 3.3 Input Validation & Error Handling

The demo refuses bad input **before** anything can crash:

| Check | Limit | Error Message |
|-------|-------|---------------|
| Empty input | — | "Enter a SMILES string or pick a molecule to start." |
| SMILES too long | 200 chars | "Molecule too large for the demo (limit 100 heavy atoms)." |
| Too many atoms | 100 heavy atoms | Same message |
| Invalid SMILES | — | "Couldn't read that SMILES. Check brackets and ring numbers, e.g. c1ccccc1." |
| Missing checkpoint | — | Shows a one-liner pointing to `make train` |

## 3.4 Demo Caveats to Say Out Loud

1. **Predictions for famous drugs can be off** even if they're in the training set (caffeine
   is one). Early stopping picks the best validation epoch, not the one that memorises
   training molecules.
2. **"Closest training molecules"** uses cosine similarity of the 256-d fingerprint, not
   structural similarity. It shows what **the model** considers alike.
3. **Toxicity probabilities run high** because of pos_weight. Compare molecules with each
   other, don't read 0.8 as "80% risk".

---

# PART 4: LIMITATIONS

## 4.1 Known Limitations (Be Ready to Discuss Each)

### 1. GCNConv Ignores Bond Types
- A C=O (double bond) and a C–O (single bond) send **identical messages**.
- Bond type, conjugation, and ring membership are already stored in `edge_attr`.
- **Fix:** Replace GCNConv with an edge-aware layer:
  - **GINE:** message = MLP([h_neighbour, e_bond])
  - **GATv2 with edge_dim:** attention weights informed by bond features
  - **D-MPNN:** message-passing on bonds instead of atoms
- This would be a change to **`model.py` only** — the featuriser already stores everything
  needed.

### 2. ESOL Is Small
- Only ~1,100 molecules, ~114 in the test set.
- One scaffold group (the PCBs) can move the RMSE by more than the difference between models.
- Seed-to-seed spread is large.

### 3. No Hyperparameter Tuning
- Hyperparameters were fixed in advance for both models.
- A tuned RF (or one with count-based fingerprints instead of bits) would probably close
  some of the ESOL gap.

### 4. Tox21 Probabilities Are Not Calibrated
- The pos_weight inflates predicted probabilities.
- The demo shows them for **ranking molecules**, not as absolute risk.

### 5. Three Seeds ≠ Significance Test
- Three runs give a rough error bar, not a formal statistical test.
- For proper significance: more seeds + a paired test (e.g., paired t-test or Wilcoxon).

---

# PART 5: FUTURE WORK

## 5.1 Concrete Next Steps (Ranked by Impact)

### 1. Edge-Aware Message Passing (Highest Priority)
> "Replace GCNConv with GINE or GATv2 with `edge_dim`, so the model can distinguish single
> from double bonds. Our featuriser already stores bond type, conjugation, and ring membership.
> We'd only change `model.py`. This should help especially on conjugated systems where
> bond order matters for solubility and reactivity."

### 2. Deterministic DeepChem Scaffold Split
> "Compare against DeepChem's deterministic split, where the rarest scaffolds go to test.
> This would let us place our Tox21 numbers alongside published results."

### 3. Saliency Validation by Deletion
> "Mask the top-k salient atoms and measure how much the prediction moves vs masking
> random atoms. If saliency is meaningful, the prediction should move more when salient
> atoms are masked."

### 4. Additional Ideas
- **Larger datasets:** BBBP (blood-brain barrier), BACE, HIV from MoleculeNet.
- **Attention visualisation:** GATv2 would give interpretable attention weights per edge.
- **Calibration:** Platt scaling or temperature scaling for Tox21 probabilities.
- **Uncertainty quantification:** Monte Carlo dropout or ensembles.

---

# PART 6: COMPARISON TABLE — GCN vs ECFP+RF (Your Summary Slide)

| Aspect | GCN (Learned FP) | ECFP4 + Random Forest |
|--------|-------------------|----------------------|
| Fingerprint | 256-d, task-specific, learned | 2048-bit, fixed, task-agnostic |
| Bond awareness | No (GCNConv limitation) | Yes (included in hash) |
| Atom counting | Yes (mean pool + mass) | No (bit = present or not) |
| Can extrapolate | Somewhat (linear head) | No (bounded by training range) |
| Explainability | Atom-level saliency maps | Feature importance only |
| Training time | ~1 min/seed (ESOL, CPU) | Seconds |
| Complexity | ~87K parameters | 500 trees, no gradients |
| ESOL RMSE | **1.063 ± 0.163** | 1.694 ± 0.498 |
| Tox21 ROC-AUC | **0.811 ± 0.009** | 0.756 ± 0.015 |
| Wins on Tox21 assays | **11/12** | 1/12 (SR-ARE) |

---

# POSSIBLE QUESTIONS FOR RAKESH

### Q1: "Why use ECFP as the baseline instead of another GNN?"
→ "ECFP + random forest is the standard baseline in molecular property prediction papers.
It's well-established, fast, and requires no deep learning. Comparing a GNN against it
tests whether the added complexity of learning a fingerprint is justified. For a GNN-vs-GNN
comparison (e.g., GCN vs GAT vs MPNN), we'd need more computational budget and it would be
a different research question — 'which GNN architecture' rather than 'does learning a
fingerprint help'."

### Q2: "Can the RF ever beat the GCN?"
→ "Yes, in specific cases. On Tox21 SR-ARE, the RF actually beats the GCN (0.746 vs 0.734).
A tuned RF with count-based fingerprints (instead of binary bits) might also close the ESOL
gap. The GCN's advantage is structural — it learns task-specific features and can capture
atom-count information — but the RF is simpler, faster, and has fewer failure modes."

### Q3: "What would you change if you had more time?"
→ "Three things, in order: (1) Replace GCNConv with an edge-aware layer like GINE or GATv2
to use bond features we already store. (2) Compare against DeepChem's deterministic scaffold
split so our Tox21 numbers are directly comparable with papers. (3) Validate the saliency
maps with a deletion test — mask top-k salient atoms and check if the prediction moves more
than masking random atoms."

### Q4: "Why does the demo warn against reading toxicity probabilities as risk?"
→ "Because we use pos_weight to handle class imbalance. Only 7.5% of labelled entries are
positive, so we upweight them during training. This makes the model take toxic examples more
seriously, but it also inflates the predicted probabilities. A molecule with a predicted
probability of 0.8 is not '80% likely to be toxic' — it just scores higher than one
predicted at 0.3. For absolute risk you'd need calibration (Platt scaling or temperature
scaling), which we didn't implement."

### Q5: "How does the 'closest training molecules' feature work?"
→ "At startup, the app computes the 256-dimensional learned fingerprint for every training
molecule using `model.embed()`. When a new molecule is entered, we compute its fingerprint
and find the 3 training molecules with the highest cosine similarity. This shows what the
model considers structurally similar — which may differ from traditional chemical
similarity."

### Q6: "Is this ready for real drug discovery?"
→ "No. It's a research demo. The model wasn't validated against experimental measurements
beyond ESOL and Tox21. Tox21 probabilities are uncalibrated. The saliency maps are
approximate. ESOL is tiny and results depend on the split. For real use you'd need more
diverse training data, proper calibration, external validation, and regulatory review.
The footer says 'Research demo. Not for safety or medical decisions.'"

---

# QUICK REFERENCE CARD

| Item | Value |
|------|-------|
| Baseline fingerprint | ECFP4 (Morgan radius 2, 2048 bits) |
| Baseline model | RandomForestRegressor/Classifier, 500 trees |
| Tox21 RF approach | 1 forest per assay, trained on labelled rows only |
| Ablation: depth | 1, 2, 3, 4 layers → all within noise |
| Ablation: readout | mean, max, mean+max → all within noise |
| Ablation: split | scaffold vs random → random is leaky (clear effect) |
| Demo app | Streamlit at localhost:8501 |
| Demo molecules | 8 presets + custom SMILES |
| Input limits | 200 chars, 100 heavy atoms |
| #1 limitation | GCNConv ignores bond types |
| #1 future work | Edge-aware layer (GINE/GATv2) |
| Key code files | `baseline.py`, `evaluate.py` (ablation path), `app/streamlit_app.py` |
