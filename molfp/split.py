"""Seeded Bemis-Murcko scaffold split, saved to data/splits/ (invariants 4 and 8).

Algorithm (docs/adr/002):
1. Group molecules by scaffold SMILES (acyclic molecules share the empty scaffold "").
2. Groups larger than half the val/test size go first (they would unbalance val/test),
   then the rest in a seeded random order.
3. Deal groups greedily: into train while it stays <= 80%, else val while <= 10%, else test.
A whole group always lands in one split, so no scaffold crosses splits.
"""

import hashlib
import json
import random
from collections import defaultdict
from functools import cache

from rdkit.Chem.Scaffolds import MurckoScaffold

from molfp.data import ROOT, load_dataset
from molfp.featurize import parse_smiles

SPLIT_DIR = ROOT / "data" / "splits"
FRACS = (0.8, 0.1, 0.1)


@cache
def scaffold_of(smiles):
    mol, _ = parse_smiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(smiles, seed):
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[scaffold_of(s)].append(i)

    n = len(smiles)
    train_cap, val_cap = FRACS[0] * n, FRACS[1] * n
    big = [g for g in groups.values() if len(g) > val_cap / 2]
    small = [g for g in groups.values() if len(g) <= val_cap / 2]
    rng = random.Random(seed)
    rng.shuffle(big)
    rng.shuffle(small)

    train, val, test = [], [], []
    for g in big + small:
        if len(train) + len(g) <= train_cap:
            train += g
        elif len(val) + len(g) <= val_cap:
            val += g
        else:
            test += g
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def random_split(n, seed):
    """Ablation only: ignores scaffolds, so analogues leak across splits."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    a, b = int(FRACS[0] * n), int((FRACS[0] + FRACS[1]) * n)
    return {"train": sorted(idx[:a]), "val": sorted(idx[a:b]), "test": sorted(idx[b:])}


def index_hash(split):
    blob = json.dumps([split["train"], split["val"], split["test"]]).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def get_split(dataset, seed, kind="scaffold"):
    """Load data/splits/{dataset}[_random]_seed{n}.json, creating it on first use.

    Both the GCN and the RF baseline call this, so they always see the same indices.
    """
    name = (
        f"{dataset}_seed{seed}.json" if kind == "scaffold" else f"{dataset}_random_seed{seed}.json"
    )
    path = SPLIT_DIR / name
    if path.exists():
        split = json.loads(path.read_text())
        if index_hash(split) != split["hash"]:
            raise ValueError(f"{path} was edited: stored hash does not match its indices")
        if split["n"] != len(load_dataset(dataset)[0]):
            raise ValueError(f"{path} was made for {split['n']} molecules; data has changed")
        return split

    smiles = load_dataset(dataset)[0]
    if kind == "scaffold":
        split = scaffold_split(smiles, seed)
    elif kind == "random":
        split = random_split(len(smiles), seed)
    else:
        raise ValueError(f"unknown split kind {kind!r}")
    split.update(dataset=dataset, seed=seed, kind=kind, n=len(smiles), hash=index_hash(split))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split))
    return split
