"""Hand-written training loop for MolGCN.

    python -m molfp.train --config configs/esol_gcn.yaml --seed 0

Model selection uses the val split only (early stopping on val RMSE / val mean ROC-AUC).
The test split is never touched here (invariant 5); molfp.evaluate scores it once.
"""

import argparse
import csv
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from molfp.data import ROOT, load_graphs
from molfp.featurize import ATOM_DIM, FEATURIZER_VERSION
from molfp.metrics import masked_roc_auc, regression_metrics
from molfp.model import MolGCN
from molfp.split import get_split

RUNS = ROOT / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    dataset: str
    name: str = "gcn"  # run folder: runs/{dataset}/{name}/seed{n}
    split: str = "scaffold"
    hidden: int = 128
    layers: int = 3
    dropout: float = 0.2
    readout: str = "meanmax"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 30


def load_config(path, **overrides):
    raw = yaml.safe_load(Path(path).read_text()) | overrides
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    return Config(**raw)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def target_stats(y, train_idx):
    """ESOL normalization stats from the train split only (invariant 6)."""
    t = y[train_idx]
    return {"mean": float(t.mean()), "std": float(t.std())}


def masked_bce(logits, y, pos_weight=None):
    """BCE over labelled entries only; NaN labels are dropped, never filled (invariant 7)."""
    m = ~torch.isnan(y)
    pw = None if pos_weight is None else pos_weight.expand_as(y)[m]
    return F.binary_cross_entropy_with_logits(logits[m], y[m], pos_weight=pw)


def tox21_pos_weight(y_train):
    """Per task: (#negatives / #positives) over labelled train rows, to offset imbalance."""
    pos = np.nansum(y_train, axis=0)
    neg = np.sum(~np.isnan(y_train), axis=0) - pos
    return torch.tensor(neg / np.maximum(pos, 1), dtype=torch.float)


def make_loss(dataset, y_all, train_idx):
    """Return (loss_fn(out, y), norm stats or None)."""
    if dataset == "esol":
        s = target_stats(y_all, train_idx)
        return (lambda out, y: F.mse_loss(out, (y - s["mean"]) / s["std"])), s
    pw = tox21_pos_weight(y_all[train_idx]).to(DEVICE)
    return (lambda out, y: masked_bce(out, y, pw)), None


def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(batch), batch.y)
        loss.backward()
        opt.step()
        total += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total / n


@torch.no_grad()
def predict(model, graphs, norm=None, batch_size=256):
    """Predictions in original units: logS for ESOL (norm given), probabilities for Tox21."""
    model.eval()
    outs = [model(b.to(DEVICE)).cpu() for b in DataLoader(graphs, batch_size=batch_size)]
    out = torch.cat(outs).numpy()
    return out * norm["std"] + norm["mean"] if norm else 1 / (1 + np.exp(-out))


def score(dataset, y_true, pred):
    """(metric dict, selection value where higher is better)."""
    if dataset == "esol":
        m = regression_metrics(y_true, pred)
        return m, -m["rmse"]
    m = masked_roc_auc(y_true, pred)
    return m, m["roc_auc"]


def build_model(cfg, out_dim):
    return MolGCN(ATOM_DIM, out_dim, cfg.hidden, cfg.layers, cfg.dropout, cfg.readout)


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    if ckpt["featurizer_version"] != FEATURIZER_VERSION:
        raise ValueError(f"{path} was trained with featurizer v{ckpt['featurizer_version']}")
    cfg = Config(**ckpt["config"])
    model = build_model(cfg, ckpt["out_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def run_dir(cfg, seed, root=RUNS):
    return Path(root) / cfg.dataset / cfg.name / f"seed{seed}"


def train(cfg, seed, root=RUNS, verbose=True):
    set_seed(seed)
    graphs = load_graphs(cfg.dataset)
    split = get_split(cfg.dataset, seed, cfg.split)
    y_all = torch.cat([g.y for g in graphs]).numpy()
    train_g = [graphs[i] for i in split["train"]]
    val_g = [graphs[i] for i in split["val"]]
    y_val = y_all[split["val"]]

    loss_fn, norm = make_loss(cfg.dataset, y_all, split["train"])
    model = build_model(cfg, y_all.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        train_g,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    out = run_dir(cfg, seed, root)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    val_name = "val_rmse" if cfg.dataset == "esol" else "val_roc_auc"
    writer.writerow(["epoch", "train_loss", val_name])

    best, best_epoch = -np.inf, 0
    for epoch in range(1, cfg.max_epochs + 1):
        loss = train_epoch(model, loader, opt, loss_fn)
        if not np.isfinite(loss):
            raise RuntimeError(f"train loss became {loss} at epoch {epoch}")
        metrics, sel = score(cfg.dataset, y_val, predict(model, val_g, norm))
        writer.writerow([epoch, f"{loss:.5f}", f"{abs(sel):.5f}"])
        log.flush()  # a crash still leaves a readable partial log
        if sel > best:
            best, best_epoch = sel, epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "seed": seed,
                    "out_dim": y_all.shape[1],
                    "norm": norm,
                    "featurizer_version": FEATURIZER_VERSION,
                    "split_hash": split["hash"],
                    "epoch": epoch,
                    "val": metrics,
                },
                out / "best.pt",
            )
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  loss {loss:.4f}  {val_name} {abs(sel):.4f}")
        if epoch - best_epoch >= cfg.patience:
            break
    log.close()
    if verbose:
        print(
            f"[train] {cfg.dataset}/{cfg.name} seed {seed}: best {val_name} "
            f"{abs(best):.4f} at epoch {best_epoch} -> {out / 'best.pt'}"
        )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train MolGCN for one seed")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(load_config(args.config), args.seed)
