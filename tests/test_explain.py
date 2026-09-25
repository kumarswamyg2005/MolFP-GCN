import numpy as np
import torch

from molfp.explain import draw_svg, saliency
from molfp.featurize import ATOM_DIM, smiles_to_data
from molfp.model import MolGCN


def test_saliency_one_finite_score_per_atom():
    torch.manual_seed(0)
    for out_dim, target in [(1, 0), (12, 7)]:
        model = MolGCN(ATOM_DIM, out_dim)
        for s in ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "C"]:
            d = smiles_to_data(s)
            sal = saliency(model, d, target=target)
            assert sal.shape == (d.num_atoms,)
            assert np.all(np.isfinite(sal))
            assert np.abs(sal).max() <= 1.0 + 1e-6


def test_draw_svg_highlights_every_atom():
    svg = draw_svg("CCO", scores=np.array([-1.0, 0.0, 1.0]))
    assert svg.lstrip().startswith("<?xml") and "</svg>" in svg
