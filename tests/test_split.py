import json

import pytest

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

    # with a consistent hash, the file wins over recomputation (proves it is read, not rebuilt)
    on_disk["hash"] = split_mod.index_hash(on_disk)
    path.write_text(json.dumps(on_disk))
    assert get_split("esol", seed=0)["test"] == on_disk["test"]
