# Architecture

MolFP-GCN is a local research codebase plus a demo app. There is no server, no database and
no network access after the one-time data download. "Scale" here means one laptop CPU and
two small datasets (ESOL ≈ 1.1k molecules, Tox21 ≈ 7.8k).

## Requirements

Functional
- Turn a SMILES string into a molecular graph (atoms = nodes, bonds = edges).
- Train a GCN that maps the graph to a 256-d learned fingerprint and then to a property:
  ESOL log-solubility (regression) or 12 Tox21 assays (multi-task classification).
- Train a fixed-fingerprint baseline (ECFP4 + random forest) on the exact same splits.
- Report every metric as mean ± std over seeds 0, 1, 2; regenerate every figure from JSON.
- Explain a prediction with a per-atom saliency score.
- Demo: SMILES in → drawing, predictions, saliency, nearest training molecules out.

Non-functional
- Reproducible: one `make all` on a clean clone, deterministic splits, seeded training.
- Explainable: every module is small enough to walk through in a viva.
- Cheap: main results in under 30 min on a CPU. CUDA is used if present, never required.
- Robust: invalid or oversized input never crashes anything (invariants 3 and 12).

## Components

```
 scripts/download_data.py ──► data/raw/*.csv  (+ data/CHECKSUMS.txt)
                                   │
                                   ▼
 molfp/data.py        load CSV, parse each SMILES, drop + count invalid rows
                                   │  list[str], targets (NaN = missing)
                                   ▼
 molfp/split.py       Bemis–Murcko scaffold split 80/10/10, seeded
                                   │  data/splits/{dataset}_seed{n}.json  ◄── shared contract
                  ┌────────────────┴──────────────────┐
                  ▼                                   ▼
 molfp/featurize.py  SMILES → graph           molfp/baseline.py  SMILES → ECFP4 bits
   x [atoms × 35], edge_index [2 × 2·bonds]      2048-bit Morgan r=2
                  │                                   │
                  ▼                                   ▼
 molfp/model.py   MolGCN                         RandomForest (1 per task)
   Linear → 3×GCNConv → mean‖max pool
          → 256-d learned fingerprint (embed)
          → MLP head → 1 or 12 outputs
                  │
 molfp/train.py   hand-written loop, early stop on val → runs/{ds}/gcn/seed{n}/best.pt
                  │                                   │
                  └──────────────┬────────────────────┘
                                 ▼
 molfp/evaluate.py   test metrics once per run → results/{esol,tox21,ablations}.json
                                 │
                 ┌───────────────┼──────────────────────┐
                 ▼               ▼                      ▼
 molfp/figures.py        molfp/explain.py          app/streamlit_app.py
 reports/figures/*.png   grad × input saliency     loads seed-0 checkpoints,
                         → RDKit SVG highlight     featurize → predict → explain
```

## Data flow for one molecule

1. `featurize.parse_smiles("CCO")` → RDKit `Mol` or `(None, reason)`.
2. `featurize.smiles_to_data` → PyG `Data(x, edge_index, edge_attr, y, smiles, num_atoms)`.
   Each bond becomes two directed edges; hydrogens are a feature, not nodes.
3. `MolGCN.embed(batch)` → three rounds of neighbour averaging, then mean and max over atoms
   → 256 numbers. This vector is the learned fingerprint.
4. `MolGCN.forward(batch)` → head on the fingerprint → logS (normalized) or 12 logits.
5. Inference inverts ESOL normalization using the train-split mean/std saved in `best.pt`,
   and applies a sigmoid to Tox21 logits.
6. `explain.saliency` → ∂output/∂x · x summed over features → one score per atom in [-1, 1].

## File contracts

| File | Written by | Read by | Content |
|---|---|---|---|
| `data/raw/*.csv` | download_data | data | untouched source CSVs |
| `data/splits/{ds}_seed{n}.json` | split | train, baseline, evaluate, app | `{"train": [...], "val": [...], "test": [...]}` row indices + hash |
| `runs/{ds}/gcn/seed{n}/best.pt` | train | evaluate, explain, app | state_dict, config, norm stats, featurizer version, split hash |
| `runs/{ds}/gcn/seed{n}/log.csv` | train | figures | epoch, train loss, val metric |
| `results/*.json` | evaluate | figures, report | per-seed metrics + mean/std, test predictions for parity plots |

The split file is the single source of truth for "which molecule is in which set". The GCN
and RF never compute their own split (invariant 8).

## Key decisions and trade-offs

| Decision | Chosen | Cost we accept | Revisit when |
|---|---|---|---|
| GNN layer | GCNConv (ADR 001) | ignores bond types | move to GAT/MPNN with edge features |
| Split | scaffold (ADR 002) | lower, noisier scores than random | never for the headline number |
| Features | own featurizer (ADR 003) | more code to own and test | — |
| Training loop | hand-written | no free logging/checkpoint helpers | many more experiments |
| Config | YAML → dataclass | no sweeps | hyperparameter search matters |
| Demo | Streamlit | limited layout control | a real product UI is needed |
| Storage | files + JSON | no experiment DB | results outgrow a folder |

## Failure handling

- Invalid SMILES: `parse_smiles` returns `(None, reason)`. The loader drops the row and logs
  the count; the app shows the DESIGN.md error copy.
- Oversized input: the app rejects more than 200 characters or more than 100 heavy atoms
  before featurizing.
- Missing checkpoint: the app shows one line pointing to `make train`.
- Download failure: fall back to `torch_geometric.datasets.MoleculeNet` for the raw CSV.
- NaN loss or divergence: stop and debug; don't patch around it.

## What I would change as it grows

- Edge-aware message passing (bond type matters for solubility and reactivity).
- A results store keyed by config hash instead of fixed JSON filenames.
- Batch inference for the demo's nearest-neighbour index if the training set gets large
  (today the 256-d training embeddings are computed once at startup, ~1k × 256).
