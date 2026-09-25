# ADR-002: Bemis–Murcko scaffold split for all headline numbers

**Status:** Accepted
**Date:** 2026-09-25
**Deciders:** Kumar (author), course evaluator

## Context
Molecular datasets contain series of close analogues: the same ring system with a different
substituent. With a random split, near-copies of test molecules end up in training, and the
score measures memorisation of a chemical series more than generalisation to new chemistry.
The viva panel will ask how we know there is no leakage.

## Decision
Split every dataset 80/10/10 by Bemis–Murcko scaffold (RDKit `MurckoScaffold`), so no scaffold
appears in more than one split. The split is seeded, saved to
`data/splits/{dataset}_seed{n}.json`, and loaded by both the GCN and the RF baseline. A random
split exists only as a labelled ablation.

Procedure (seeded, "balanced" variant): group molecules by scaffold SMILES. Scaffold groups
larger than half the test size go to train first, since putting one in val or test would
unbalance the split. The remaining groups are shuffled with the seed and dealt greedily:
train until 80% is full, then val until 90%, then test.

## Options considered

### Option A: Random split
**Pros:** simplest; highest numbers; matches older papers.
**Cons:** leaks analogue series; overstates real-world accuracy.

### Option B: Deterministic scaffold split (sorted by group size, no seed)
**Pros:** standard in DeepChem; fully deterministic.
**Cons:** only one possible split, so "3 seeds" only varies model initialisation, not which
chemistry is held out. The test set ends up as the smallest, rarest scaffolds.

### Option C: Seeded balanced scaffold split (chosen)
**Pros:** no scaffold leakage; each seed holds out a different set of scaffolds, so the std
across seeds includes split variance, which is the honest error bar.
**Cons:** numbers are lower and noisier than random split; not identical to any single
published benchmark split.

### Option D: Time or cluster split
**Pros:** even stricter.
**Cons:** no timestamps in ESOL/Tox21; clustering adds a parameter we would have to defend.

## Trade-off analysis
We give up a flattering number for an honest one. The random-split ablation shows the size of
the gap, which is itself a result worth reporting.

## Consequences
- Easier: answering "how do you know there's no leakage": a test asserts zero scaffold
  overlap.
- Harder: comparing directly with random-split numbers in papers; the report must say which
  split each cited number used.
- Revisit: if a dataset with timestamps is added, consider a time split.

## Action items
1. [ ] `split.py` with a test for zero overlap, determinism, and 80/10/10 within 1%.
2. [ ] Random-split ablation in `results/ablations.json`, clearly labelled.
