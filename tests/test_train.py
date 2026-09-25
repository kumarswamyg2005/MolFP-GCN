import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from molfp.featurize import ATOM_DIM
from molfp.model import MolGCN
from molfp.train import make_loss, masked_bce, predict, set_seed, target_stats, train_epoch


def test_masked_bce_ignores_nan():
    logits = torch.tensor([[0.3, -1.2, 2.0], [0.5, 0.1, -0.4]])
    y = torch.tensor([[1.0, float("nan"), 0.0], [float("nan"), 1.0, float("nan")]])
    m = ~torch.isnan(y)
    expected = F.binary_cross_entropy_with_logits(logits[m], y[m])
    assert torch.isclose(masked_bce(logits, y), expected)
    # changing a masked position's logit must not change the loss
    logits2 = logits.clone()
    logits2[0, 1] = 100.0
    assert torch.isclose(masked_bce(logits2, y), expected)


def test_normalization_train_only_and_round_trip():
    y = np.array([[1.0], [2.0], [3.0], [100.0], [-50.0]], dtype=np.float32)
    train_idx = [0, 1, 2]
    s = target_stats(y, train_idx)
    y_other = y.copy()
    y_other[3:] = 0.0  # val/test values must not affect the stats
    assert s == target_stats(y_other, train_idx)
    assert np.isclose(s["mean"], 2.0)
    z = (y - s["mean"]) / s["std"]
    assert np.allclose(z * s["std"] + s["mean"], y)


def test_overfit_32_esol_molecules(need_data):
    from molfp.data import load_graphs

    set_seed(0)
    graphs = load_graphs("esol")[:32]
    y = torch.cat([g.y for g in graphs]).numpy()
    loss_fn, norm = make_loss("esol", y, list(range(32)))
    model = MolGCN(ATOM_DIM, 1, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(graphs, batch_size=32)
    for _ in range(300):  # 1 batch per epoch -> 300 steps
        train_epoch(model, loader, opt, loss_fn)
    rmse = np.sqrt(np.mean((predict(model, graphs, norm) - y) ** 2))
    assert rmse < 0.1, rmse
