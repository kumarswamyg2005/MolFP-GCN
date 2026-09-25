"""Load the raw CSVs, drop invalid SMILES (logged), build graphs."""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from molfp.featurize import parse_smiles, smiles_to_data

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]  # fmt: skip

DATASETS = {
    "esol": {"file": "delaney-processed.csv", "tasks": ["measured log solubility in mols per litre"]},
    "tox21": {"file": "tox21.csv", "tasks": TOX21_TASKS},
}


@cache
def load_dataset(name):
    """Return (smiles list, targets float32 [n, n_tasks] with NaN = missing, task names).

    Row i here is index i in the split files. Invalid SMILES are dropped and the count logged.
    """
    spec = DATASETS[name]
    df = pd.read_csv(RAW / spec["file"])
    ok = df["smiles"].map(lambda s: parse_smiles(s)[0] is not None)
    if (~ok).sum():
        print(f"[data] {name}: dropped {(~ok).sum()} of {len(df)} rows with invalid SMILES")
    df = df[ok].reset_index(drop=True)
    y = df[spec["tasks"]].to_numpy(dtype=np.float32)
    return df["smiles"].tolist(), y, spec["tasks"]


@cache
def load_graphs(name):
    """Graphs for every valid row, in the same order as load_dataset."""
    smiles, y, _ = load_dataset(name)
    return [smiles_to_data(s, yi) for s, yi in zip(smiles, y, strict=True)]
