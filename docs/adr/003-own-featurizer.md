# ADR-003: Our own atom/bond featurizer instead of PyG's MoleculeNet features

**Status:** Accepted
**Date:** 2026-09-25
**Deciders:** Kumar (author), course evaluator

## Context
PyG's `MoleculeNet` dataset class ships precomputed graphs (`x` of 9 integer-coded atom
features from OGB's `from_smiles`). They are convenient, but they're integer indices meant for
embedding tables, the exact choices are defined in another library, and the demo must
featurize arbitrary user SMILES the same way as training.

## Decision
Write `molfp/featurize.py`: RDKit parses the SMILES, and each atom becomes a 35-d vector of
concatenated one-hots (element, degree, formal charge, total Hs, hybridization) plus aromatic,
in-ring and mass/100. Every feature is listed in the module docstring table. Each bond becomes
two directed edges, and hydrogens are implicit. PyG's `MoleculeNet` is only a fallback for
downloading the raw CSV.

## Options considered

### Option A: PyG built-in (`MoleculeNet` / `from_smiles`)
| Dimension | Assessment |
|---|---|
| Complexity | Lowest: zero code |
| Explainability | Low: features live in someone else's code; integer codes need an embedding |
| Consistency | Risk: demo input path differs from training path unless we also call `from_smiles` |

### Option B: DeepChem `ConvMolFeaturizer`
| Dimension | Assessment |
|---|---|
| Complexity | Low code, heavy dependency |
| Explainability | Medium: 75-d features, many we'd have to justify |
| Consistency | Adds a large, version-sensitive dependency to the locked stack |

### Option C: Own featurizer (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | ~100 lines |
| Explainability | High: every column is documented and tested |
| Consistency | One function, `smiles_to_data`, used by training, evaluation, explain and the demo |

## Trade-off analysis
About 100 lines of code buy full control and a single code path from CSV to demo. The features
are the standard Duvenaud/DeepChem-style set, so we lose no expressiveness that matters here.

## Consequences
- Easier: saliency is interpretable per feature group; the viva question "what's in an atom
  vector?" has a table for an answer.
- Harder: we own the bugs. Tests cover benzene, ethanol, invalid input, and a constant dim.
- Revisit: a `FEATURIZER_VERSION` constant is saved in each checkpoint; bump it on any change
  so stale checkpoints are rejected rather than silently mis-featurized.

## Action items
1. [ ] `featurize.py` with docstring table and `FEATURIZER_VERSION`.
2. [ ] Tests 1–4.
