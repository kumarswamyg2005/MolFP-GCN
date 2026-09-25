import pytest

from app.streamlit_app import COPY, PRESETS, solubility_band, validate
from molfp.featurize import smiles_to_data


def test_presets_featurize():
    assert len(PRESETS) == 8
    for name, s in PRESETS.items():
        assert smiles_to_data(s) is not None, name
        assert validate(s)[1] is None, name


@pytest.mark.parametrize(
    "smiles, message",
    [
        ("", COPY["empty"]),
        ("   ", COPY["empty"]),
        ("C1CC", COPY["invalid"]),
        ("xyz", COPY["invalid"]),
        ("<script>alert(1)</script>", COPY["invalid"]),
        ("C" * 201, COPY["too_long"]),
        ("C" * 101, COPY["too_large"]),
    ],
)
def test_app_rejects_bad_input(smiles, message):
    mol, err = validate(smiles)
    assert mol is None and err == message


def test_app_accepts_limits():
    assert validate("C" * 100)[1] is None  # exactly 100 heavy atoms is allowed


def test_solubility_bands():
    assert solubility_band(-1.0) == "soluble"
    assert solubility_band(-2.0) == "soluble"
    assert solubility_band(-3.0) == "moderately soluble"
    assert solubility_band(-4.5) == "poorly soluble"


def test_tox21_preset_is_an_unseen_positive(need_data):
    import numpy as np

    from molfp.data import load_dataset
    from molfp.split import get_split

    smiles, y, _ = load_dataset("tox21")
    i = smiles.index(PRESETS["Tox21 positive (test set)"])
    assert i in get_split("tox21", 0)["test"]
    assert np.nansum(y[i]) >= 1
