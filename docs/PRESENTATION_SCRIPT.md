# Presentation script: MolFP-GCN (about 6 minutes)

Speakers: **K** = Kumaraswamy, **S** = Simon Peter, **R** = Rakesh. Slide numbers match
`MolFP-GCN_presentation.pdf`. Say it in your own words; the numbers are the part to get exactly right.

---

## Slide 1: Title (K)
"Good morning, sir. We are Kumaraswamy, Simon Peter and Rakesh. Our project is **MolFP-GCN**:
learning molecular fingerprints with graph convolutional networks. We'll show how a computer can
predict properties of a molecule from its structure, and how it can tell us why."

## Slide 2: The question (K)
"We asked one question: can a computer look only at a molecule's structure and predict two things?
First, **solubility**, how well it dissolves in water. That matters for medicines, because a drug that
doesn't dissolve can't reach the blood. We used the ESOL dataset, 1,128 molecules.
Second, **toxicity** in 12 lab tests, from the Tox21 dataset with 7,823 molecules. Tox21 is messy:
17% of the answers are missing and only 7.5% of known answers say 'toxic'.
And third: when the model guesses, can it show **which atoms** it used? That's our explanation part."

## Slide 3: Two ways to describe a molecule (K)
"A computer can't learn from text, so a molecule must become numbers. There are two ways.
The **old way** is the ECFP fingerprint. A fixed rule looks at each atom and its neighbours and ticks boxes in
a 2048-long checklist. It is the same whatever you want to predict. We give it to a random forest;
this is our **baseline**.
**Our way** treats the molecule as a graph: atoms are nodes, bonds are edges. In each round, every atom
averages information from its neighbours. After three rounds, each atom knows about everything within three bonds.
We pool all atoms into 256 numbers. That is a **learned fingerprint**, shaped by the task.
So the central question is: does the learned fingerprint beat the fixed one?"

## Slide 4: Honest evaluation (S)
"Thank you. I'll explain how we kept the test fair. Chemistry datasets contain families of near-identical molecules.
If one is in training and its twin is in test, the test is too easy. So we use a **scaffold split**:
each family, defined by its core ring structure, stays together in one group. The test set is
made of structures the model has never seen. We use 80% train, 10% validation, 10% test, and the
test set is used once, at the very end.
We repeat everything with **three seeds** and report mean and spread.
To prove it matters, we also tried a random split: the error dropped from 1.06 to 0.80. That looks better, but it's
leakage, not skill. For Tox21 we ignore missing labels instead of calling them safe, and we give extra weight to the rare toxic labels."

## Slide 5: Results (S)
"Here are the results on the test set, mean ± standard deviation over three seeds.
On ESOL, the GCN gets RMSE **1.063** against **1.694** for ECFP plus random forest, and R-squared of 0.709 against 0.267.
On Tox21, mean ROC-AUC is **0.811** against **0.756**, and the GCN is ahead on **11 of the 12** assays.
One honest caveat: the ESOL gap depends on which scaffolds land in the test set. In one seed, a group of
chlorinated molecules all fell in the test set and hurt the forest badly. So the direction is clear, but the exact
margin varies. Rakesh will show the figures."

## Slide 6: What the model learned (R)
"On the left is the solubility parity plot: predicted against measured. The GCN points sit closer to the diagonal line
than the random forest's. On the right is ROC-AUC for each of the 12 Tox21 assays. Blue is the GCN, orange is the
forest. The GCN is higher on every assay except SR-ARE."

## Slide 7: Explainability and demo (R)
"This is our demo app, built in Streamlit. You draw or paste a molecule. It predicts solubility and
the 12 toxicity scores, and it colours the atoms that pushed a prediction up or down, using saliency.
You start it with `make demo`, or open a link like `?smiles=CCO` for ethanol."
*(If allowed, run the live demo here.)*

## Slide 8: Limitations and next steps (R)
"We want to be upfront. Our GCN layer ignores bond types, so a double bond looks like a single bond to it.
The next step is an edge-aware layer such as GINE or D-MPNN. ESOL is small, about 1,100 molecules, so results vary between seeds.
Our scaffold split may also be easier than the one used in papers, so we shouldn't compare our numbers directly with theirs.
On the positive side, everything is reproducible: one `make all` command takes about 11 minutes on a laptop and
gives identical results."

## Slide 9: Who did what (K)
"To summarise our roles. I built the GCN model, the featurisation from SMILES to graph, and the saliency.
Simon built the data pipeline, the scaffold split, training and evaluation. Rakesh built the baseline,
the ablations, the figures and the demo app."

## Slide 10: Thank you (all)
"In short: a learned fingerprint beat a fixed one on both tasks, tested honestly. Thank you, sir. We'd be happy to take questions."

---

## Likely questions (short answers)
- **Why a scaffold split?** Stops near-duplicate molecules leaking between train and test.
- **Why is ESOL's spread so large?** Only about 1,100 molecules; a few scaffold groups change the score a lot.
- **Why ROC-AUC for Tox21?** Toxic labels are rare, so accuracy would be misleading.
- **What does the GCN do that ECFP doesn't?** Its fingerprint is learned for the task instead of fixed.
- **Biggest weakness?** GCNConv ignores bond types.
