<!-- generated from reports/REPORT.src.md by python -m molfp.figures; edit the source -->
# Learned molecular fingerprints with a graph convolutional network

| Name | Roll no. |
|---|---|
| Gurram Naga Kumaraswamy | S20230010101 |
| Simon Peter | S20230010171 |
| Rakesh | S20230010196 |

Course: `<COURSE NAME AND CODE>` · 2026

All numbers in this report are filled in by `python -m molfp.figures` from `results/*.json`.
None are typed by hand. Every result is the mean ± sample standard deviation over seeds
0, 1 and 2, where the seed changes both the scaffold split and the model initialisation.

## 1. Introduction

A molecular fingerprint turns a molecule into a fixed-length vector that a standard model can
use. The usual choice is ECFP (a Morgan fingerprint): each atom's neighbourhood, up to some
radius, is hashed into one of 2048 bits. The hash is fixed. Aspirin gets the same bits
whether the question is solubility or liver toxicity.

A graph convolutional network (GCN) does something similar but learns it. Atoms are nodes,
bonds are edges, and each layer lets an atom mix in information from its bonded neighbours.
After a few layers the atom vectors are pooled into one molecule vector, and the whole thing
is trained end to end, so the "fingerprint" ends up shaped by the task. Duvenaud et al. (2015)
called this a differentiable fingerprint.

We wanted to test two things on public data. First, does the learned fingerprint actually beat
ECFP + random forest when both get the same molecules and the same honest split? Second, can
the model point at the atoms behind a prediction?

## 2. Data

| Dataset | Task | Molecules used | Test molecules (seed 0) |
|---|---|---|---|
| ESOL (Delaney) | log solubility in mol/L, regression | 1,128 | 114 |
| Tox21 | 12 toxicity assays, binary, labels partly missing | 7,823 | 783 |

Both come from the DeepChem mirror. Their SHA-256 hashes are in `data/CHECKSUMS.txt`, and
the download script refuses to continue if a file changes. RDKit cannot sanitise
8 of the Tox21 SMILES, so they are dropped with a logged count; ESOL loses none. 17% of Tox21
labels are missing. They are masked out of the loss and out of every ROC-AUC, never filled with 0.

**Split.** Molecules are grouped by Bemis–Murcko scaffold (the ring systems and the linkers
between them) and whole groups are assigned to train, validation or test in an 80/10/10
ratio, so no scaffold appears in more than one set. Scaffold groups bigger than half the test
size go to train first; the rest are shuffled with the seed. The index files live in
`data/splits/` and both models read the same file. A test checks that the GCN and the random
forest follow even a deliberately altered split file. Another checks that no molecule
(compared as canonical SMILES) sits in both train and test.

## 3. Method

**Featurisation.** Each atom is a 35-number vector: one-hot element (C, N, O, S, F, Cl, Br, I,
P, other), degree, formal charge, hydrogen count, hybridisation, plus aromatic and in-ring
flags and mass/100. Each bond becomes two directed edges. Hydrogens are counted as an atom
feature rather than added as nodes. Bond types are stored but the GCN does not use them
(see Limitations).

**Model.** A linear layer maps atoms to 128 dimensions. Three GCNConv layers follow, each with
batch norm, ReLU and dropout 0.2, and residual connections from the second layer on. One
GCNConv layer computes

  H′ = D^−½ (A + I) D^−½ H W

which means each atom takes a degree-normalised average of itself and its neighbours and then
applies a shared linear map. Mean pooling and max pooling over atoms are concatenated into a
256-d vector. This is the learned fingerprint. A two-layer MLP maps it to one output (logS)
or twelve logits. Mean and max do not depend on atom order, and a test confirms that
renumbering the atoms changes the output by less than 1e-5.

**Baseline.** ECFP4 (Morgan radius 2, 2048 bits) with a 500-tree random forest. For Tox21
there is one forest per assay, trained only on the molecules labelled for that assay.

**Training.** Adam (lr 1e-3, weight decay 1e-5), batch 64, up to 300 epochs, early stopping
with patience 30 on the validation metric (RMSE for ESOL, mean ROC-AUC for Tox21). ESOL
targets are standardised with the training-set mean and standard deviation only. Tox21 uses
masked binary cross-entropy with a per-assay positive weight (negatives/positives in the
training set), because only 7.5% of labels are positive. The test set is scored once,
after training, by `molfp.evaluate`. As a check on that claim, one test scrambles the
features and labels of every test molecule, retrains, and asserts that the weights come out
bit-identical.

## 4. Results

| Model | ESOL RMSE ↓ | ESOL MAE ↓ | ESOL R² ↑ | Tox21 mean ROC-AUC ↑ |
|---|---|---|---|---|
| GCN (learned fingerprint) | 1.063 ± 0.163 | 0.856 ± 0.124 | 0.709 ± 0.048 | 0.811 ± 0.009 |
| ECFP4 + random forest | 1.694 ± 0.498 | 1.287 ± 0.378 | 0.267 ± 0.248 | 0.756 ± 0.015 |

The GCN has the lower error on ESOL and the higher ROC-AUC on Tox21. On Tox21 the gap is
larger than the seed-to-seed spread. The GCN is ahead on 11 of the 12
assays; the forest is ahead on SR-ARE.

The ESOL comparison needs a caveat. The random forest's standard deviation is very large, and
the per-seed numbers show why:

| Seed | GCN RMSE | RF RMSE | GCN ROC-AUC | RF ROC-AUC |
|---|---|---|---|---|
| 0 | 0.904 | 1.288 | 0.803 | 0.756 |
| 1 | 1.230 | 2.249 | 0.810 | 0.741 |
| 2 | 1.055 | 1.545 | 0.820 | 0.772 |

In seed 1 the whole biphenyl scaffold group, which includes the polychlorinated biphenyls, fell
into the test set. These are the least soluble molecules in ESOL (logS down to -11.6).
A random forest predicts by averaging training targets, so it cannot go below the range it
was trained on. ECFP also only records that a chlorine environment is present, not how many
there are. For the least soluble test molecule the forest predicted -4.6 and the GCN
-7.2. The forest's RMSE for that seed is 2.25. The GCN got 1.23 on the same split. It
sees atom counts through mean pooling and the mass feature, and its linear head can
extrapolate a little. On seeds 0 and 2, which are more typical, the forest scores
1.29 and 1.55 against the GCN's 0.90 and
1.06. So the GCN still wins, but by less than the headline mean suggests.

![ESOL parity plots](figures/parity_esol.png)

Two results fall outside the sanity bands we set before training. The ESOL forest (RMSE
1.694 ± 0.498) is above the expected 0.9–1.4, for the reason above. The Tox21 GCN (ROC-AUC
0.811 ± 0.009) is slightly above the expected 0.68–0.80. We checked for duplicates between
train and test and found none. Our best explanation is the split variant. DeepChem's
deterministic scaffold split sends the rarest scaffolds to test, while this one sends the
large groups to train and samples the test set from the remaining small groups at random.
That is likely an easier test set than the one behind most published Tox21 scaffold numbers,
so these numbers should not be compared directly with papers.

![Tox21 per-assay ROC-AUC](figures/tox21_per_task.png)

| Task | GCN | ECFP4 + RF |
|---|---|---|
| NR-AR | 0.858 ± 0.051 | 0.818 ± 0.053 |
| NR-AR-LBD | 0.882 ± 0.050 | 0.878 ± 0.057 |
| NR-AhR | 0.839 ± 0.014 | 0.783 ± 0.024 |
| NR-Aromatase | 0.763 ± 0.046 | 0.756 ± 0.025 |
| NR-ER | 0.781 ± 0.021 | 0.687 ± 0.045 |
| NR-ER-LBD | 0.836 ± 0.026 | 0.745 ± 0.066 |
| NR-PPAR-gamma | 0.817 ± 0.035 | 0.724 ± 0.050 |
| SR-ARE | 0.734 ± 0.024 | 0.746 ± 0.022 |
| SR-ATAD5 | 0.797 ± 0.040 | 0.732 ± 0.020 |
| SR-HSE | 0.802 ± 0.013 | 0.703 ± 0.058 |
| SR-MMP | 0.835 ± 0.013 | 0.773 ± 0.015 |
| SR-p53 | 0.784 ± 0.011 | 0.726 ± 0.055 |

Training curves are in `figures/training_curves.png`. Validation RMSE on ESOL is noisy from
epoch to epoch (the seed-0 validation set has 112 molecules), and the best epochs were
24, 43, 25. Tox21 validation AUC is smoother, with best epochs 73, 28, 56.

## 5. Ablations (ESOL)

![ESOL ablations](figures/ablations_esol.png)

| Variant | GCN RMSE | RF RMSE |
|---|---|---|
| depth 1 | 1.082 ± 0.204 | not run |
| depth 2 | 1.002 ± 0.135 | not run |
| depth 3 | 1.063 ± 0.163 | not run |
| depth 4 | 1.053 ± 0.156 | not run |
| readout mean | 0.985 ± 0.172 | not run |
| readout max | 1.046 ± 0.072 | not run |
| readout meanmax | 1.063 ± 0.163 | not run |
| split scaffold | 1.063 ± 0.163 | 1.694 ± 0.498 |
| split random | 0.800 ± 0.023 | 1.256 ± 0.037 |

**Depth.** One to four layers give 1.08 ± 0.20, 1.00 ± 0.14, 1.06 ± 0.16 and
1.05 ± 0.16. Every difference is well inside one standard deviation. We can't claim that
three layers is best, or that depth matters at all at this dataset size. We expected
over-smoothing to hurt at four layers and did not see it, possibly because of the residual
connections.

**Readout.** Mean, max and mean+max give 0.98 ± 0.17, 1.05 ± 0.07 and
1.06 ± 0.16. Again no clear winner. We kept mean+max because it was the planned
default, not because the ablation favoured it.

**Split.** This is the one clear effect. With a random split the GCN reaches
0.80 ± 0.02 and the forest 1.26 ± 0.04, against 1.06 ± 0.16
and 1.69 ± 0.50 under the scaffold split. A random split puts close analogues of
test molecules in the training set, and both models look better than they are. This is the
reason every headline number here uses the scaffold split.

## 6. What the fingerprint learned

![t-SNE of ECFP bits vs learned fingerprint](figures/tsne_pair.png)

The t-SNE plots show the ESOL test molecules under each representation, coloured by measured
solubility. ECFP groups molecules by shared substructures, and solubility is mixed within its
clusters. In the learned fingerprint the least soluble molecules form their own group and the
colour changes more smoothly across the map. t-SNE distorts distances and depends on its
random seed, so we read this as a picture that agrees with the numbers, not as evidence on its
own.

**Saliency.** For a prediction *o* and atom features *X*, each atom gets the score
Σ_f X[i,f] · ∂o/∂X[i,f] (gradient × input), scaled to [−1, 1]. Red atoms push the prediction
up and blue atoms push it down.

![Saliency examples](figures/saliency_examples.png)

On ESOL the maps are mostly chemically sensible. Carbonyl and hydroxyl oxygens raise predicted
solubility, while aromatic and chlorinated carbon cores lower it. The maps have limits. A
gradient describes a tiny change to the features, not what happens if an atom is removed.
Message passing also spreads credit onto neighbours, so a highlighted carbon may be
highlighted because of the oxygen next to it. We treat saliency as a hint about what the model
used, not as a chemical explanation.

## 7. Limitations

- GCNConv ignores bond order and aromaticity of bonds. A C=O and a C–O send identical
  messages. The bond features are already stored in `edge_attr`, so an edge-aware layer could
  use them directly.
- ESOL is small. With 114 test molecules, one scaffold group can move the RMSE by more
  than the difference between models, as seed 1 shows.
- Hyperparameters were fixed in advance and not tuned for either model. A tuned forest (or
  one with count-based fingerprints) would probably close some of the ESOL gap.
- Tox21 probabilities are not calibrated. The positive weighting inflates them, so the demo
  shows them for ranking molecules, not as absolute risk.
- Three seeds give a rough error bar, not a significance test.

## 8. Future work

- Replace GCNConv with an edge-aware layer (GINE, GATv2 with `edge_dim`, or a D-MPNN) and
  check whether bond information closes the remaining ESOL errors on conjugated systems.
- Compare against the deterministic DeepChem scaffold split as well, so the Tox21 numbers can
  be placed next to published ones.
- Test the saliency maps with a deletion check: mask the top-k atoms and measure how much the
  prediction moves compared with masking random atoms.

## Reproducing

```
make setup data train eval figures test    # main results, CPU
make ablations && make figures             # ablations and their figures
make demo                                  # the Streamlit app
```

## References

- Delaney, J. S. (2004). ESOL: estimating aqueous solubility directly from molecular structure. *J. Chem. Inf. Comput. Sci.* 44(3).
- Duvenaud, D. et al. (2015). Convolutional networks on graphs for learning molecular fingerprints. *NeurIPS*.
- Kipf, T. N. & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
- Rogers, D. & Hahn, M. (2010). Extended-connectivity fingerprints. *J. Chem. Inf. Model.* 50(5).
- Wu, Z. et al. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chemical Science* 9.
- Bemis, G. W. & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. *J. Med. Chem.* 39(15).
