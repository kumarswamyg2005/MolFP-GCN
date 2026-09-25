"""Download ESOL and Tox21 raw CSVs into data/raw/ and verify/record SHA-256.

If data/CHECKSUMS.txt already lists a file and the hash differs, stop: the upstream data
changed and the committed splits and results would no longer match.
"""

import gzip
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CHECKSUMS = ROOT / "data" / "CHECKSUMS.txt"
BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"

# local file -> (url, PyG MoleculeNet name for the fallback)
SOURCES = {
    "delaney-processed.csv": (BASE + "delaney-processed.csv", "ESOL"),
    "tox21.csv": (BASE + "tox21.csv.gz", "Tox21"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, dest):
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    if url.endswith(".gz"):
        with gzip.open(tmp) as g, open(dest, "wb") as f:
            shutil.copyfileobj(g, f)
        tmp.unlink()
    else:
        tmp.rename(dest)


def fetch_via_pyg(pyg_name, dest):
    """Fallback: let PyG download the same raw CSV. We only keep the CSV, not its features."""
    from torch_geometric.datasets import MoleculeNet

    root = RAW / "_pyg"
    MoleculeNet(str(root), pyg_name)
    shutil.copy(next((root / pyg_name.lower() / "raw").glob("*.csv")), dest)
    shutil.rmtree(root)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    known = {}
    if CHECKSUMS.exists():
        for line in CHECKSUMS.read_text().split("\n"):
            if line.strip():
                h, name = line.split()
                known[name] = h

    for name, (url, pyg_name) in SOURCES.items():
        dest = RAW / name
        if not dest.exists():
            try:
                fetch(url, dest)
            except Exception as e:
                print(f"[data] {url} failed ({e}); falling back to PyG MoleculeNet")
                fetch_via_pyg(pyg_name, dest)
        h = sha256(dest)
        if name in known and known[name] != h:
            sys.exit(f"[data] checksum mismatch for {name}: expected {known[name]}, got {h}")
        known[name] = h
        print(f"[data] {name}  {h[:12]}  ok")

    CHECKSUMS.write_text("".join(f"{h}  {n}\n" for n, h in sorted(known.items())))


if __name__ == "__main__":
    main()
