"""Fixed-fingerprint baseline: ECFP4 (Morgan r=2, 2048 bits) + random forest.

Same split files and same metrics as the GCN (invariant 8). The RF has no early stopping or
tuning, so it does not use the val split at all; the test split is scored once.
"""

import argparse
import json

import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from molfp.data import load_dataset
from molfp.featurize import parse_smiles
from molfp.metrics import masked_roc_auc, regression_metrics
from molfp.split import get_split

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def ecfp(smiles):
    """ECFP4 bit vectors, uint8 [n, 2048]. Radius 2 = each bit sees atoms up to 2 bonds away."""
    return np.stack([_MORGAN.GetFingerprintAsNumPy(parse_smiles(s)[0]) for s in smiles])


def run_rf(dataset, seed, split_kind="scaffold"):
    smiles, y, tasks = load_dataset(dataset)
    split = get_split(dataset, seed, split_kind)
    X = ecfp(smiles)
    tr, te = np.array(split["train"]), np.array(split["test"])

    if dataset == "esol":
        rf = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
        rf.fit(X[tr], y[tr, 0])
        pred = rf.predict(X[te])[:, None]
        metrics = regression_metrics(y[te], pred)
    else:
        pred = np.full((len(te), len(tasks)), np.nan)
        for t in range(len(tasks)):
            lab = tr[~np.isnan(y[tr, t])]  # train only on rows labelled for this task
            rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
            rf.fit(X[lab], y[lab, t])
            pred[:, t] = rf.predict_proba(X[te])[:, 1]
        metrics = masked_roc_auc(y[te], pred)

    return {"metrics": metrics, "test_idx": te.tolist(), "test_pred": pred.tolist(),
            "split_hash": split["hash"]}  # fmt: skip


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ECFP4 + RF baseline, test metrics for one seed")
    p.add_argument("--dataset", choices=["esol", "tox21"], required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(json.dumps(run_rf(args.dataset, args.seed)["metrics"], indent=2))
