from pathlib import Path

import pytest

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture
def need_data():
    """Skip tests that need the downloaded CSVs (run `make data` first)."""
    if not (RAW / "delaney-processed.csv").exists() or not (RAW / "tox21.csv").exists():
        pytest.skip("raw data missing: run `make data`")
