<!-- generated from docs/EXPLAIN.src.md by python -m molfp.figures; edit the source -->
# EXPLAIN: viva cheat sheet

Short answers in plain words, then the detail. Numbers come from `results/*.json` through
`python -m molfp.figures`.

## The one-sentence version

ECFP hashes every atom's neighbourhood into the same bits whatever the task. The GCN learns
its own neighbourhood summary for each task, which beat ECFP + random forest here
(ESOL RMSE 1.063 ± 0.163 vs 1.694 ± 0.498; Tox21 ROC-AUC 0.811 ± 0.009 vs
0.756 ± 0.015), and gradients show which atoms it leaned on.

## Pipeline in five steps

1. RDKit parses the SMILES. Invalid input returns `None` plus a reason and never raises.
2. Every atom becomes 35 numbers; every bond becomes two directed edges (i→j, j→i).
   Hydrogens are a count on each atom, not nodes.
3. Three GCN layers. Each atom averages itself and its neighbours, then a shared linear map
   is applied.
4. Mean and max over atoms, concatenated: 256 numbers. **That vector is the learned
   fingerprint** (`model.embed`).
5. A small MLP maps it to logS (one number) or 12 toxicity logits.

## Likely questions

### 1. What does one GCNConv layer compute? Why self-loops and degree normalisation?

H′ = D^−½ (A + I) D^−½ H W

- A is the adjacency matrix (1 if two atoms are bonded). H holds one row per atom.
- **+I (self-loop):** without it an atom's new vector would contain only its neighbours and
  lose its own identity. Adding I keeps the atom in its own average.
- **D^−½ … D^−½:** the (i, j) term is divided by √(deg_i · deg_j). Without this, a carbon
  with four neighbours would sum four messages and a terminal oxygen one, so the scale of the
  vectors would depend on degree. The symmetric form keeps values stable and treats both ends
  of a bond the same way.
- **W** is one weight matrix shared by all atoms, much as a CNN kernel is shared across
  pixels. That's why it works for molecules of any size.

### 2. How is this fingerprint different from ECFP? What does "differentiable" mean?

ECFP runs the same kind of neighbourhood aggregation, but combines neighbours with a fixed
hash and writes bits. Nothing is learned, and the same molecule always gets the same bits.
In the GCN every step is a smooth function (averages, matrix products, ReLU, mean/max), so
the gradient of the loss flows back through the fingerprint into W. Training moves the
fingerprint toward whatever the task needs. That's the "differentiable" part.

### 3. Why is 3 layers roughly comparable to ECFP4? What happens with too many?

After L layers an atom's vector includes atoms up to L bonds away. ECFP4 has radius 2, so it
sees 2 bonds. Three layers see one bond further, which is in the same range. With many layers
every atom ends up averaging over nearly the whole molecule, so all atom vectors look alike.
This is **over-smoothing**, and max pooling then has nothing to pick out. In our depth
ablation, 1–4 layers were all within noise (1.08 ± 0.20, 1.00 ± 0.14,
1.06 ± 0.16, 1.05 ± 0.16). We didn't see over-smoothing at 4 layers. The residual
connections probably help, and ESOL is too small to separate these settings.

### 4. Why a scaffold split? What did the random-split ablation show?

Datasets contain series of near-copies (the same ring system, a different side chain). A
random split puts some of each series in train and some in test, so the model gets tested on
molecules it has nearly seen. The scaffold split keeps each ring-system family in exactly
one set, which is closer to the real use case: predicting new chemistry.

Ablation on ESOL: random split GCN 0.80 ± 0.02, RF 1.26 ± 0.04;
scaffold split GCN 1.06 ± 0.16, RF 1.69 ± 0.50. Both models look
clearly better on the random split, and that improvement is the leakage.

### 5. Why mean + max pooling? Why must the readout be permutation-invariant?

A molecule's atoms have no natural order; SMILES order is arbitrary. If the readout depended
on order, the same molecule written two ways would get two predictions. Sum, mean and max
ignore order, and a test renumbers atoms and checks the output moves by < 1e-5.
Mean says "what is the molecule made of on average" and max says "is there at least one atom
of this kind". A single nitro group can matter a lot and would get diluted by a mean. In the
ablation the three readouts were within noise (0.98 ± 0.17 mean,
1.05 ± 0.07 max, 1.06 ± 0.16 both).

### 6. Tox21 missing labels and class imbalance?

- Missing (17% of labels): masked. The loss is computed only on labelled
  (molecule, assay) pairs and each assay's ROC-AUC uses only its labelled molecules. Filling
  with 0 would teach the model that untested molecules are non-toxic.
- Imbalance (only 7.5% positive): `pos_weight` = negatives/positives per assay,
  computed on the training set only. Side effect: predicted probabilities run high, so the
  demo tells the viewer to compare molecules, not to read the probabilities as risk.
- The random forest baseline trains one forest per assay on that assay's labelled rows only.

### 7. The GCN ignores bond types. How would GAT or an MPNN fix that?

GCNConv's message from j to i is just j's features times a degree weight, so a C=O and a C–O
contribute the same. Fixes:
- **Edge-aware MPNN / GINE / D-MPNN:** the message is a function of the neighbour *and* the
  bond features, e.g. m_ij = MLP([h_j, e_ij]).
- **GAT / GATv2:** learns an attention weight per edge, so neighbours can matter unequally;
  with `edge_dim` the bond features feed into that weight.
Our featurizer already stores bond type, conjugation and ring membership in `edge_attr`, so
this would be a change to `model.py` only.

### 8. What does the saliency map show? When can it mislead?

Score per atom = Σ over features of (feature value × gradient of the output w.r.t. that
feature), scaled to [−1, 1]. Red means that atom's features push the prediction up; blue
means down. It is a first-order estimate. It can mislead because:
- gradients are local: they describe a tiny change, not deleting the atom;
- ReLUs saturate, so an important atom can show a small gradient;
- message passing spreads credit, so a carbon may light up because of the oxygen next to it;
- it explains the *model*, which may have learned a shortcut, not the chemistry.

### 9. Did the GCN beat the RF? If not, why, and is that a failure?

Yes, on both tasks: ESOL RMSE 1.063 ± 0.163 vs 1.694 ± 0.498, Tox21 ROC-AUC
0.811 ± 0.009 vs 0.756 ± 0.015 (GCN ahead on 11 of 12 assays; RF ahead
on SR-ARE).

Be ready for the caveat. The RF's ESOL average is dragged down by seed 1 (RMSE
2.25), where the polychlorinated biphenyls all landed in test. A random forest
can't predict below its training range, and ECFP bits don't count chlorines. On the other two
seeds the gap is smaller: RF 1.29 / 1.55 vs GCN
0.90 / 1.06. Losing would not have been a failure either;
the honest result on a fixed protocol is the point.

### 10. How do you know there's no leakage?

- Scaffold split: a test asserts zero scaffold overlap between train/val/test for every seed.
- No duplicate molecules across train and test, by canonical SMILES (tested).
- Normalisation stats use train rows only (tested: changing val/test targets doesn't change
  them).
- Early stopping uses validation only. The strongest test scrambles every test molecule's
  features and labels to garbage, retrains, and asserts the weights are **bit-identical**.
  If the test set were used anywhere in training, they would differ. We checked the test
  itself by scrambling training molecules instead, and then it fails as it should.
- GCN and RF read the same hashed split file; evaluation raises an error if their hashes
  differ.

## Numbers to remember

| Model | ESOL RMSE ↓ | ESOL MAE ↓ | ESOL R² ↑ | Tox21 mean ROC-AUC ↑ |
|---|---|---|---|---|
| GCN (learned fingerprint) | 1.063 ± 0.163 | 0.856 ± 0.124 | 0.709 ± 0.048 | 0.811 ± 0.009 |
| ECFP4 + random forest | 1.694 ± 0.498 | 1.287 ± 0.378 | 0.267 ± 0.248 | 0.756 ± 0.015 |

Tox21 scored 0.811 ± 0.009, slightly above our expected band. Our explanation: our
balanced scaffold split samples test scaffolds at random, which is easier than DeepChem's
"rarest scaffolds go to test" split, so don't compare it directly with papers.

## Demo caveats to say out loud

- Predictions for famous drugs can be noticeably off even when they are in the training set
  (caffeine is one). Early stopping picks the epoch with the best validation error, so the
  model does not memorise its training molecules.
- "Closest training molecules" uses cosine similarity of the ESOL model's 256-d fingerprint,
  not structural similarity. It shows what the model considers alike.
- The demo refuses SMILES over 200 characters or 100 heavy atoms, and anything RDKit can't
  parse.
