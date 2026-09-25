"""Score GCN and RF on the test split, 3 seeds each, and write results/*.json.

    python -m molfp.evaluate --dataset esol      -> results/esol.json
    python -m molfp.evaluate --dataset tox21     -> results/tox21.json
    python -m molfp.evaluate --ablations         -> results/ablations.json

A GCN run is reused if its checkpoint exists and was trained on the current split (hash
match); otherwise it is trained here. Each run's test split is predicted exactly once, here
(invariant 5). Every number is reported as mean ± sample std over seeds 0, 1, 2 (invariant 9).
"""

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch
import yaml

from molfp.baseline import run_rf
from molfp.data import ROOT, load_dataset, load_graphs
from molfp.invariants import SEEDS
from molfp.split import get_split
from molfp.train import load_config, load_model, predict, run_dir, score, train

RESULTS = ROOT / "results"
MAIN_CONFIG = {"esol": "configs/esol_gcn.yaml", "tox21": "configs/tox21_gcn.yaml"}


def run_gcn(cfg, seed):
    split = get_split(cfg.dataset, seed, cfg.split)
    ckpt_path = run_dir(cfg, seed) / "best.pt"
    old = torch.load(ckpt_path, weights_only=True) if ckpt_path.exists() else None
    if old and old["split_hash"] == split["hash"] and old["config"] == asdict(cfg):
        print(f"[eval] reusing {ckpt_path}")
    else:
        train(cfg, seed)
    model, ckpt = load_model(ckpt_path)

    graphs = load_graphs(cfg.dataset)
    te = split["test"]
    pred = predict(model, [graphs[i] for i in te], ckpt["norm"])
    y = load_dataset(cfg.dataset)[1]
    metrics, _ = score(cfg.dataset, y[te], pred)
    return {
        "metrics": metrics,
        "test_idx": te,
        "test_pred": pred.tolist(),
        "best_epoch": ckpt["epoch"],
        "split_hash": split["hash"],
    }


def summarize(runs):
    """mean ± sample std (ddof=1) over seeds for every scalar or per-task metric."""
    out = {}
    for key in runs[0]["metrics"]:
        vals = np.array([r["metrics"][key] for r in runs], dtype=float)
        out[key] = {
            "mean": np.nanmean(vals, axis=0).tolist(),
            "std": np.nanstd(vals, axis=0, ddof=1).tolist(),
        }
    return out


def evaluate_dataset(dataset):
    cfg = load_config(MAIN_CONFIG[dataset])
    smiles, y, tasks = load_dataset(dataset)
    result = {
        "dataset": dataset,
        "split": cfg.split,
        "seeds": list(SEEDS),
        "tasks": tasks,
        "config": asdict(cfg),
    }
    for name, fn in [("gcn", lambda s: run_gcn(cfg, s)), ("rf", lambda s: run_rf(dataset, s))]:
        runs = []
        for seed in SEEDS:
            print(f"[eval] {dataset} {name} seed {seed}")
            r = fn(seed)
            runs.append({"seed": seed, **r})
        result[name] = {"summary": summarize(runs), "per_seed": runs}
    for g, r in zip(result["gcn"]["per_seed"], result["rf"]["per_seed"], strict=True):
        if g["split_hash"] != r["split_hash"]:  # invariant 8
            raise RuntimeError(f"seed {g['seed']}: GCN and RF used different splits")
    # true test targets per seed, so figures never need to reload data
    for r in result["gcn"]["per_seed"]:
        r["test_true"] = y[r["test_idx"]].tolist()
    write(f"{dataset}.json", result)
    print_summary(result)


def evaluate_ablations(path="configs/esol_ablations.yaml"):
    spec = yaml.safe_load((ROOT / path).read_text())
    base = load_config(spec["base"])
    out = {"dataset": base.dataset, "seeds": list(SEEDS), "variants": {}}
    for variant, overrides in spec["variants"].items():
        cfg = load_config(spec["base"], **overrides)
        if cfg != base:  # identical-to-main variants reuse the main runs
            cfg = load_config(spec["base"], name=f"abl_{variant}", **overrides)
        runs = []
        for seed in SEEDS:
            print(f"[ablation] {variant} seed {seed}")
            runs.append({"seed": seed, **run_gcn(cfg, seed)})
        entry = {
            "overrides": overrides,
            "gcn": summarize(runs),
            "gcn_per_seed": [{"seed": r["seed"], **r["metrics"]} for r in runs],
        }
        if "split" in overrides:  # the RF on the same split shows the gap is not GCN-specific
            rf = [{"seed": s, **run_rf(base.dataset, s, cfg.split)} for s in SEEDS]
            entry["rf"] = summarize(rf)
        out["variants"][variant] = entry
    write("ablations.json", out)
    for v, e in out["variants"].items():
        print(f"  {v:18s} GCN RMSE {e['gcn']['rmse']['mean']:.3f} ± {e['gcn']['rmse']['std']:.3f}")


def write(name, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))
    print(f"[eval] wrote results/{name}")


def print_summary(result):
    keys = ["rmse", "mae", "r2"] if result["dataset"] == "esol" else ["roc_auc"]
    for model in ("gcn", "rf"):
        s = result[model]["summary"]
        print(
            f"  {model.upper():4s} "
            + "  ".join(f"{k} {s[k]['mean']:.3f} ± {s[k]['std']:.3f}" for k in keys)
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test-set evaluation over 3 seeds")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=["esol", "tox21"])
    g.add_argument("--ablations", action="store_true")
    args = p.parse_args()
    evaluate_ablations() if args.ablations else evaluate_dataset(args.dataset)
