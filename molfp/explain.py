"""Gradient x input saliency: which atoms pushed this prediction up or down.

For output o and atom feature matrix X [atoms x 35]:
    score_i = sum_f  X[i, f] * d o / d X[i, f]
i.e. a first-order (Taylor) estimate of how much atom i's features contribute to o.
Scores are divided by the largest |score|, so they lie in [-1, 1]:
    > 0  the atom raises the output (higher logS / higher toxicity logit)
    < 0  the atom lowers it.

Limits: gradients are local (they describe a tiny change, not deleting the atom), they can
saturate through ReLUs, and neighbouring atoms share credit because message passing mixes
them. Treat the map as a hint about what the model used, not as chemistry ground truth.
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from molfp.featurize import parse_smiles

BLUE = np.array([0x2F, 0x6F, 0xB0]) / 255  # lowers the prediction
RED = np.array([0xB8, 0x41, 0x2C]) / 255  # raises the prediction


def saliency(model, data, target=0):
    """One score per atom in [-1, 1] for output column `target`."""
    model.eval()
    device = next(model.parameters()).device
    x = data.x.to(device).clone().requires_grad_(True)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    out = model.head(model.pool(model.atom_embed(x, data.edge_index.to(device)), batch))
    out[0, target].backward()
    s = (x.grad * x).sum(dim=1).detach().cpu().numpy()
    peak = np.abs(s).max()
    return s / peak if peak > 0 else s


def diverging(v):
    """[-1, 1] -> RGB on blue -> white -> red."""
    end = RED if v > 0 else BLUE
    a = min(abs(float(v)), 1.0)
    return tuple((1 - a) * np.ones(3) + a * end)


def draw_svg(smiles, scores=None, size=(460, 400)):
    """RDKit SVG of the molecule, atoms tinted by saliency if scores are given."""
    return _draw(rdMolDraw2D.MolDraw2DSVG(*size), smiles, scores)


def draw_png(smiles, scores=None, size=(460, 400)):
    """Same drawing as PNG bytes (for matplotlib report figures)."""
    return _draw(rdMolDraw2D.MolDraw2DCairo(*size), smiles, scores)


def _draw(d, smiles, scores):
    mol, _ = parse_smiles(smiles)
    opts = d.drawOptions()
    opts.clearBackground = True
    opts.padding = 0.12
    opts.bondLineWidth = 2
    kwargs = {}
    if scores is not None:
        atoms = list(range(mol.GetNumAtoms()))
        kwargs = {
            "highlightAtoms": atoms,
            "highlightAtomColors": {i: diverging(scores[i]) for i in atoms},
            "highlightAtomRadii": {i: 0.45 for i in atoms},
            "highlightBonds": [],
        }
    rdMolDraw2D.PrepareAndDrawMolecule(d, Chem.Mol(mol), **kwargs)
    d.FinishDrawing()
    return d.GetDrawingText()
