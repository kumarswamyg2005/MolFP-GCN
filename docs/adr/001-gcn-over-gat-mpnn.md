# ADR-001: GCNConv as the message-passing layer

**Status:** Accepted
**Date:** 2026-09-25
**Deciders:** Kumar (author), course evaluator

## Context
The project claim is that a graph network can learn a task-specific fingerprint that competes
with a fixed ECFP fingerprint. It will be defended in a viva, so every layer must be explainable
from first principles. It trains on a laptop CPU, and the data is small (ESOL ≈ 1.1k
molecules), which makes overfitting a bigger risk than lack of capacity.

## Decision
Use PyTorch Geometric `GCNConv` (Kipf & Welling, 2017) with 3 layers, hidden size 128 and
mean‖max readout. Store bond features in `edge_attr` but don't feed them to the model.

## Options considered

### Option A: GCNConv
| Dimension | Assessment |
|---|---|
| Complexity | Low: one normalized adjacency product per layer |
| Cost | Cheapest per epoch |
| Explainability | High: H' = σ(D^-1/2 (A+I) D^-1/2 H W) fits on one line |
| Fit to data | Few parameters, less overfitting on 1.1k molecules |

**Pros:** closed-form math, no edge-feature plumbing, fast, the direct analogue of ECFP's
"look r bonds away".
**Cons:** treats all bonds the same, so a C=O and a C–O look identical to the message passing.
Neighbour weights are fixed by degree, not learned.

### Option B: GAT (graph attention)
| Dimension | Assessment |
|---|---|
| Complexity | Medium: attention coefficients per edge, multiple heads |
| Cost | ~2–3× GCN per epoch |
| Explainability | Medium: attention weights are tempting but not faithful explanations |
| Fit to data | More parameters on a small dataset |

**Pros:** learns which neighbours matter; can use edge features in PyG's `GATv2Conv`.
**Cons:** more to explain and tune; gains on ESOL-sized data are usually within seed noise.

### Option C: Edge-aware MPNN (Gilmer et al., 2017) / D-MPNN
| Dimension | Assessment |
|---|---|
| Complexity | High: edge network, message and update functions |
| Cost | Highest |
| Explainability | Lower in a 10-minute viva slot |
| Fit to data | State of the art on MoleculeNet, but needs tuning |

**Pros:** uses bond type and aromaticity directly; strongest published results.
**Cons:** most code and hyperparameters; harder to defend line by line.

## Trade-off analysis
The project is about the *idea* of a learned fingerprint versus a fixed one, not about
squeezing out the best score. GCN is the smallest model that shows that idea, and its update
rule maps cleanly onto ECFP's radius. What it costs us is bond information. That is a known,
named limitation, and GAT or an MPNN would fix it.

## Consequences
- Easier: the math section of EXPLAIN.md is short; training finishes in minutes on CPU.
- Harder: the GCN may lose to ECFP+RF on ESOL, where bond order matters. If it does, the report
  says so and points here.
- Revisit: when adding edge features. `edge_attr` is already built, so switching to
  `GINEConv` or `GATv2Conv(edge_dim=...)` only changes `model.py`.

## Action items
1. [x] Build bond features into `edge_attr` in `featurize.py` even though GCN ignores them.
2. [ ] State the limitation in README and REPORT.md.
