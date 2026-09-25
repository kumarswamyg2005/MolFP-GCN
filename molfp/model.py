"""MolGCN: graph convolutions over atoms -> 256-d learned fingerprint -> property head.

What one GCNConv layer computes
-------------------------------
    H' = D^-1/2 (A + I) D^-1/2  H  W

- H is the [atoms x features] matrix, A the adjacency matrix (A[i, j] = 1 if bonded).
- A + I adds a self-loop, so each atom keeps its own features in the average.
- D^-1/2 ... D^-1/2 divides the (i, j) term by sqrt(deg_i * deg_j): a carbon with four
  neighbours doesn't get four times the signal of a terminal oxygen.
- W is a learned weight matrix shared by every atom (like a CNN kernel shared across pixels).
In words: every atom takes a degree-normalised average of itself and its bonded neighbours,
then applies the same linear transform. We then apply BatchNorm, ReLU and dropout.

Why depth ~ ECFP radius
-----------------------
After 1 layer an atom's vector mixes in atoms 1 bond away; after L layers, atoms up to L bonds
away. ECFP4 (radius 2) hashes each atom's 2-bond neighbourhood into bits with a fixed hash.
Here, a learned, differentiable function takes the place of that hash. With L = 3 each atom
sees one bond further than ECFP4. Too many layers over-smooth: every atom's vector converges
to the same molecule-wide average (see the depth ablation).

Readout
-------
concat(mean over atoms, max over atoms) -> 256-d. Mean and max don't depend on atom order,
so the whole model is permutation-invariant (invariant 10). Mean captures composition; max
captures "is there at least one atom/group like this" (e.g. one nitro group).
This vector is the learned fingerprint: `model.embed(batch)`.
"""

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class MolGCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, layers=3, dropout=0.2, readout="meanmax"):
        super().__init__()
        self.readout = readout
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(GCNConv(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        fp_dim = 2 * hidden if readout == "meanmax" else hidden
        self.head = nn.Sequential(
            nn.Linear(fp_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
        )

    def atom_embed(self, x, edge_index):
        """Per-atom vectors after L rounds of message passing."""
        h = self.inp(x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            out = self.dropout(torch.relu(norm(conv(h, edge_index))))
            # residual from layer 2 on: layer 1 changes the representation most, later
            # layers refine it, and the skip keeps gradients healthy as depth grows
            h = h + out if i >= 1 else out
        return h

    def pool(self, h, batch):
        if self.readout == "mean":
            return global_mean_pool(h, batch)
        if self.readout == "max":
            return global_max_pool(h, batch)
        return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)

    def embed(self, data):
        """The learned fingerprint: [num_molecules, 256]."""
        return self.pool(self.atom_embed(data.x, data.edge_index), data.batch)

    def forward(self, data):
        """[num_molecules, out_dim]: normalized logS (ESOL) or 12 logits (Tox21)."""
        return self.head(self.embed(data))
