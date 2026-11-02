# Viva Script — Kumaraswamy (S20230010101)

## Your Focus Areas
1. **GCN Model Architecture & Theory** (the core ML)
2. **Featurisation Pipeline** (SMILES → graph)
3. **Saliency / Explainability**

> Read [00_OVERVIEW.md](00_OVERVIEW.md) first for the project-level context everyone shares.

---

# PART 1: THE GCN MODEL — WHAT IT IS AND HOW IT WORKS

## 1.1 What Is a Graph Convolutional Network?

**Opening line for the viva:**
> "A GCN is a neural network that operates on graph-structured data. In our case, each
> molecule is a graph — atoms are nodes, bonds are edges — and the GCN learns to combine
> atom information through message passing to produce a molecule-level representation."

**How it works in simple steps:**
1. Each atom starts with a **35-dimensional feature vector** (element type, degree, charge,
   hydrogen count, hybridisation, aromaticity, ring membership, mass).
2. A linear layer projects those 35 numbers into **128 dimensions**.
3. Then **3 GCNConv layers** run. In each layer, every atom takes a **degree-normalised
   average** of its own features and its bonded neighbours' features, then applies a shared
   linear transformation.
4. After all layers, **mean pooling** (average across all atoms) and **max pooling** (max
   across all atoms) are concatenated → a **256-dimensional learned fingerprint**.
5. A 2-layer MLP maps this fingerprint to the final prediction: 1 number (logS for ESOL)
   or 12 logits (Tox21 toxicity).

**Diagram to draw on the board:**
```
Atom features [n_atoms × 35]
       │
  Linear(35 → 128)
       │
  GCNConv + BN + ReLU + Dropout   ← Layer 1
       │
  GCNConv + BN + ReLU + Dropout   ← Layer 2 (+residual)
       │
  GCNConv + BN + ReLU + Dropout   ← Layer 3 (+residual)
       │
  mean ‖ max  pooling → [256]     ← THE LEARNED FINGERPRINT
       │
  MLP(256 → 128 → output)         ← Property head
```

## 1.2 The GCNConv Equation — Be Ready to Write This

$$H' = D^{-1/2} (A + I) D^{-1/2} \cdot H \cdot W$$

**Break it down piece by piece:**

| Symbol | What It Is | Why It's There |
|--------|-----------|----------------|
| **H** | Atom feature matrix [n_atoms × features] | Current state of all atoms |
| **A** | Adjacency matrix (1 if two atoms bonded) | Defines the graph structure |
| **+I** | Self-loop (identity matrix added) | Without it, an atom's update would lose its own identity — it would only contain neighbour info |
| **D** | Degree matrix (diagonal: how many bonds each atom has) | — |
| **D^{−½} … D^{−½}** | Symmetric normalisation | Without this, a carbon with 4 neighbours sums 4 messages while a terminal oxygen sums 1. The symmetric form divides by √(deg_i · deg_j), keeping values stable |
| **W** | Learnable weight matrix, shared across ALL atoms | Like a CNN kernel shared across pixels — this is what makes it work for molecules of any size |

**Analogy to use:**
> "It's like people in a room. Each person asks their neighbours what they know, averages
> it with their own knowledge, and updates their opinion. After 3 rounds, everyone knows
> about people up to 3 handshakes away."

## 1.3 Why 3 Layers? Connection to ECFP Radius

- After **L layers**, each atom's vector contains information from atoms up to **L bonds away**.
- **ECFP4** has radius 2, so it sees atoms 2 bonds away.
- **3 GCN layers** see one bond further — same neighbourhood range.
- **Too many layers** → **over-smoothing**: every atom's vector converges to the molecule-wide
  average, and max pooling has nothing to pick out.

**Ablation result (be ready to quote):**
- Depth 1: RMSE 1.082 ± 0.204
- Depth 2: RMSE 1.002 ± 0.135
- Depth 3: RMSE 1.063 ± 0.163 (our default)
- Depth 4: RMSE 1.053 ± 0.156
- All within noise — we didn't see over-smoothing at 4 layers, probably because of **residual connections** (skip connections from layer 2 onwards).

## 1.4 Residual Connections — Why and Where

**In our code (`model.py` line 56):**
```python
h = h + out if i >= 1 else out  # residual from layer 2 on
```

- Layer 1 changes the representation the most (from 35-dim atom features to abstract embeddings).
- Layers 2 and 3 refine it. The **skip connection** lets gradients flow directly to earlier layers,
  preventing vanishing gradients and reducing over-smoothing.

## 1.5 Why Mean + Max Pooling? Permutation Invariance

**Key concept:** A molecule's atoms have **no natural order**. The SMILES `c1ccccc1` and
`c1ccc(cc1)` describe the same benzene differently. If our readout depended on atom order,
the same molecule would get two different predictions.

**Mean pooling** captures: "What is this molecule made of on average?"
**Max pooling** captures: "Is there at least one atom/group like X?" (e.g., one nitro group
can dominate toxicity but would get diluted by mean).

**Test:** we renumber atoms randomly and check the output differs by < 1e-5.

**Ablation result:**
- Mean only: 0.985 ± 0.172
- Max only: 1.046 ± 0.072
- Mean+Max: 1.063 ± 0.163
- No clear winner — we kept mean+max as it was the planned default.

---

# PART 2: THE FEATURISATION PIPELINE

## 2.1 From SMILES to Graph

**Pipeline (be ready to trace through):**
```
"CC(=O)Oc1ccccc1C(=O)O"  (aspirin)
        │
   RDKit parses it
        │
   featurize.parse_smiles() → (Mol object, None) or (None, "reason")
        │
   featurize.mol_to_data() → PyG Data object:
        • x: [n_atoms × 35] float tensor — atom feature vectors
        • edge_index: [2 × 2·n_bonds] long tensor — directed edges
        • edge_attr: [2·n_bonds × 6] float tensor — bond features (STORED but NOT USED by GCNConv)
        • y: target label(s)
        • smiles: original string
        • num_atoms: count
```

## 2.2 Atom Features — 35 Numbers (KNOW THIS TABLE)

| Feature Block | Values | Width | Columns |
|---------------|--------|-------|---------|
| Element | C, N, O, S, F, Cl, Br, I, P, other | 10 | 0–9 |
| Degree | 0, 1, 2, 3, 4, 5, other | 7 | 10–16 |
| Formal charge | −1, 0, +1, other | 4 | 17–20 |
| Total H count | 0, 1, 2, 3, 4 | 5 | 21–25 |
| Hybridisation | SP, SP2, SP3, SP3D, SP3D2, other | 6 | 26–31 |
| Is aromatic | 0/1 | 1 | 32 |
| Is in ring | 0/1 | 1 | 33 |
| Mass / 100 | float (C = 0.12) | 1 | 34 |

**Why "other" slots?** If a rare element appears that's not in our list, it falls into the
"other" bin instead of crashing. Same for unusual degrees or charges.

**Why mass/100?** Raw mass (e.g. 12.0, 16.0) would dominate the other features which are 0/1.
Dividing by 100 keeps it in a similar range.

## 2.3 Bond Features — 6 Numbers (Stored But Not Used)

| Feature | Values |
|---------|--------|
| Bond type | single, double, triple, aromatic (one-hot, 4 dims) |
| Is conjugated | 0/1 |
| Is in ring | 0/1 |

**Critical talking point:** GCNConv ignores these! A C=O and a C–O send identical messages.
The features are stored in `edge_attr` so a future edge-aware layer (GINE, GATv2, D-MPNN)
could use them **without changing the featuriser** — only `model.py` would change.

## 2.4 Graph Rules

- Every bond i–j becomes **two directed edges**: i→j and j→i. This is because PyG's
  GCNConv expects directed edges.
- **Hydrogens are implicit**: counted as a feature on the heavy atom, never added as nodes.
  This keeps the graph smaller without losing information.

## 2.5 Error Handling — parse_smiles Never Crashes

```python
def parse_smiles(smiles):
    """Return (mol, None) or (None, reason). Never raises."""
```

- Empty/non-string input → `(None, "empty input")`
- Syntax error → `(None, "syntax error (check brackets and ring-closure digits)")`
- Invalid chemistry → `(None, "invalid chemistry: {e}")`
- No atoms → `(None, "no atoms")`

**This is invariant 3.** The demo shows a friendly error message; the data loader drops
and logs the count.

---

# PART 3: SALIENCY / EXPLAINABILITY

## 3.1 What Is Saliency?

For a prediction _o_ and atom feature matrix _X_ [atoms × 35]:

**Score for atom i** = Σ_f X[i,f] × ∂o/∂X[i,f]

This is a **gradient × input** attribution. It's a first-order (Taylor) estimate of how
much atom i's features contribute to the output.

**In code (`explain.py`):**
```python
out[0, target].backward()
s = (x.grad * x).sum(dim=1)  # one score per atom
# then scale to [-1, 1] by dividing by max |score|
```

## 3.2 How to Read the Colours

| Colour | Meaning |
|--------|---------|
| **Red** | This atom's features push the prediction **UP** (more soluble / more toxic) |
| **Blue** | This atom's features push the prediction **DOWN** (less soluble / less toxic) |
| **White** | Didn't matter much |

## 3.3 Chemical Sense Check — Examples to Mention

- **ESOL:** Carbonyl (C=O) and hydroxyl (O-H) oxygens → **red** (they like water, increase solubility).
  Aromatic carbon rings and chlorinated carbons → **blue** (hydrophobic, decrease solubility).
  This makes chemical sense!

## 3.4 Limitations of Saliency (IMPORTANT — Will Be Asked)

1. **Gradients are local**: they describe a tiny change to features, not what happens if you
   remove the atom entirely.
2. **ReLU saturation**: an important atom can show a small gradient if the ReLU is in its
   flat region.
3. **Message passing spreads credit**: a carbon may light up because of the oxygen bonded
   to it — they've been mixing information for 3 rounds.
4. **It explains the model, not the chemistry**: the model may have learned a shortcut or
   correlation, not the true mechanism.

**Good closing line:**
> "We treat saliency as a hint about what the model relied on, not as a chemical explanation."

---

# POSSIBLE QUESTIONS FOR KUMARASWAMY

### Q1: "What does one GCNConv layer compute?"
→ Write the equation, explain A+I (self-loop), D^{-½} (degree normalisation), W (shared weights).

### Q2: "Why not use GAT or MPNN instead?"
→ "GCNConv was our starting point because it's the simplest graph layer. Its limitation is
that it ignores bond types — a C=O and C–O send the same message. GAT learns attention
weights per edge so neighbours contribute unequally. An MPNN/GINE computes messages as a
function of both the neighbour and the bond features. Our featuriser already stores bond
type, conjugation and ring membership in `edge_attr`, so switching to an edge-aware layer
would be a change to `model.py` only — no new data processing needed. That's our top
recommendation for future work."

### Q3: "How many parameters does the model have?"
→ "Roughly:
- Input linear: 35 × 128 = 4,480
- 3 GCNConv layers: 3 × 128 × 128 ≈ 49,152
- BatchNorm: 3 × 256 = 768
- MLP head: 256 × 128 + 128 × 1 (or 12) ≈ 32,896
- Total: ~87,000 parameters. Small enough to train on CPU in under a minute per seed."

### Q4: "What is over-smoothing?"
→ "With many message-passing layers, every atom ends up averaging over nearly the whole
molecule. All atom vectors converge to the same value, so max pooling has nothing to
distinguish. We tested 1–4 layers and didn't see it, likely because residual connections
let later layers add to the original representation rather than replace it."

### Q5: "Why gradient × input instead of just gradients?"
→ "Raw gradients tell you how *sensitive* the output is to each feature, but a feature
that's zero contributes nothing regardless of its gradient. Multiplying by the feature value
weights the sensitivity by what's actually there. It's a first-order Taylor approximation of
removing that feature."

### Q6: "Could you use attention-based explanations instead?"
→ "Yes, if we used GATv2 we'd get attention weights per edge which show how much each
neighbour contributed. But attention weights can be misleading — they show what the model
attends to, not necessarily what drives the output. Gradient-based methods are model-agnostic
and more directly tied to the output value."

---

# QUICK REFERENCE CARD

| Item | Value |
|------|-------|
| Model name | MolGCN |
| Input dim | 35 (atom features) |
| Hidden dim | 128 |
| GCN layers | 3 |
| Fingerprint dim | 256 (128 mean + 128 max) |
| Dropout | 0.2 |
| Activation | ReLU |
| Normalisation | BatchNorm after each GCN layer |
| Residual | From layer 2 onwards |
| Output | 1 (ESOL logS) or 12 (Tox21 logits) |
| Bond features | Stored but NOT used (GCNConv limitation) |
| Saliency method | Gradient × input, scaled to [−1, 1] |
| Key code files | `model.py`, `featurize.py`, `explain.py` |
