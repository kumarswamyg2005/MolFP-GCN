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


def test_masked_roc_auc_ignores_nan():
    from sklearn.metrics import roc_auc_score

    from molfp.metrics import masked_roc_auc

    nan = float("nan")
    y = np.array([[1, 0], [0, nan], [1, 0], [0, 0], [nan, 0]], dtype=float)
    p = np.array([[0.9, 0.1], [0.2, 0.9], [0.6, 0.2], [0.4, 0.3], [0.99, 0.5]])
    m = masked_roc_auc(y, p)
    assert m["roc_auc_per_task"][0] == roc_auc_score([1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4])
    assert np.isnan(m["roc_auc_per_task"][1])  # task 2 has only negatives: undefined
    assert m["roc_auc"] == m["roc_auc_per_task"][0]  # undefined task excluded from the mean


def test_train_never_sees_test(need_data, tmp_path, monkeypatch):
    """Scrambling every test molecule's features and labels must not change the model."""
    import molfp.train as train_mod
    from molfp.data import load_graphs
    from molfp.split import get_split

    cfg = train_mod.load_config("configs/esol_gcn.yaml", max_epochs=3, patience=3)
    test_idx = set(get_split("esol", 0)["test"])
    clean = load_graphs("esol")
    scrambled = []
    for i, g in enumerate(clean):
        if i in test_idx:
            g = g.clone()
            g.x = torch.rand_like(g.x) * 100
            g.y = torch.full_like(g.y, 1e6)
        scrambled.append(g)

    states = []
    for graphs, name in [(clean, "a"), (scrambled, "b")]:
        monkeypatch.setattr(train_mod, "load_graphs", lambda _ds, g=graphs: g)
        out = train_mod.train(cfg, 0, root=tmp_path / name, verbose=False)
        states.append(torch.load(out / "best.pt", weights_only=True))
    assert states[0]["norm"] == states[1]["norm"]
    for k, v in states[0]["state_dict"].items():
        assert torch.equal(v, states[1]["state_dict"][k]), k


def test_config_and_checkpoint_guards(tmp_path):
    import pytest

    from molfp.train import load_config, load_model

    with pytest.raises(ValueError):
        load_config("configs/esol_gcn.yaml", learning_rate=0.1)  # typo'd key
    torch.save({"featurizer_version": "0"}, tmp_path / "old.pt")
    with pytest.raises(ValueError):
        load_model(tmp_path / "old.pt")
