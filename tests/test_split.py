import json

import pytest
import torch

from molfp import split as split_mod
from molfp.data import load_dataset
from molfp.split import get_split, scaffold_of, scaffold_split


@pytest.mark.parametrize("dataset", ["esol", "tox21"])
def test_scaffold_split_properties(need_data, dataset):
    smiles, _, _ = load_dataset(dataset)
    n = len(smiles)
    s = scaffold_split(smiles, seed=0)

    # every molecule in exactly one split
    all_idx = s["train"] + s["val"] + s["test"]
    assert sorted(all_idx) == list(range(n))

    # zero scaffold overlap
    scafs = {k: {scaffold_of(smiles[i]) for i in s[k]} for k in ("train", "val", "test")}
    assert not scafs["train"] & scafs["val"]
    assert not scafs["train"] & scafs["test"]
    assert not scafs["val"] & scafs["test"]

    # sizes within 1% of 80/10/10
    for k, frac in [("train", 0.8), ("val", 0.1), ("test", 0.1)]:
        assert abs(len(s[k]) / n - frac) <= 0.01, (k, len(s[k]) / n)

    # same seed -> identical; different seed -> different
    assert scaffold_split(smiles, seed=0) == s
    assert scaffold_split(smiles, seed=1) != s


def test_split_file_is_single_source_of_truth(need_data, tmp_path, monkeypatch):
    monkeypatch.setattr(split_mod, "SPLIT_DIR", tmp_path)
    first = get_split("esol", seed=0)
    path = tmp_path / "esol_seed0.json"
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["hash"] == first["hash"]

    # an edited file whose hash no longer matches is rejected
    on_disk["test"] = on_disk["test"][:5]
    path.write_text(json.dumps(on_disk))
    with pytest.raises(ValueError):
        get_split("esol", seed=0)

    # a split made for a different dataset size is rejected
    stale = {**on_disk, "n": on_disk["n"] + 1, "hash": split_mod.index_hash(on_disk)}
    path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="data has changed"):
        get_split("esol", seed=0)

    # with a consistent hash, the file wins over recomputation (proves it is read, not rebuilt)
    on_disk["hash"] = split_mod.index_hash(on_disk)
    path.write_text(json.dumps(on_disk))
    assert get_split("esol", seed=0)["test"] == on_disk["test"]


@pytest.mark.parametrize("dataset", ["esol", "tox21"])
def test_no_molecule_in_both_train_and_test(need_data, dataset):
    from rdkit import Chem

    smiles = load_dataset(dataset)[0]
    s = get_split(dataset, 0)
    canon = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in smiles]
    train = {canon[i] for i in s["train"]}
    assert not [canon[i] for i in s["test"] if canon[i] in train]


def test_gcn_and_rf_use_same_split(need_data, tmp_path, monkeypatch):
    from molfp.baseline import run_rf
    from molfp.train import load_config, train

    monkeypatch.setattr(split_mod, "SPLIT_DIR", tmp_path)
    real = get_split("esol", 0)
    # a deliberately odd but hash-consistent split: both models must follow it exactly
    odd = {
        "train": real["train"][:200],
        "val": real["val"],
        "test": real["test"][:20],
        "n": real["n"],
    }
    odd["hash"] = split_mod.index_hash(odd)
    (tmp_path / "esol_seed0.json").write_text(json.dumps(odd))

    rf = run_rf("esol", 0)
    assert rf["test_idx"] == odd["test"] and rf["split_hash"] == odd["hash"]
    cfg = load_config("configs/esol_gcn.yaml", max_epochs=1)
    ckpt = torch.load(train(cfg, 0, root=tmp_path, verbose=False) / "best.pt", weights_only=True)
    assert ckpt["split_hash"] == odd["hash"]
