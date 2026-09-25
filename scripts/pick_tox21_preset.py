"""Pick the demo's "Tox21 positive" preset: the seed-0 TEST molecule with the most active
assays (ties -> fewer heavy atoms, then row order), limited to <= 40 heavy atoms so it
draws legibly. Printed, then pasted into PRESETS in app/streamlit_app.py.

    python -m scripts.pick_tox21_preset
"""

import numpy as np

from molfp.data import load_dataset
from molfp.featurize import parse_smiles
from molfp.split import get_split

smiles, y, tasks = load_dataset("tox21")
cands = []
for i in get_split("tox21", 0)["test"]:
    heavy = parse_smiles(smiles[i])[0].GetNumHeavyAtoms()
    if heavy <= 40:
        cands.append((-np.nansum(y[i]), heavy, i))
_, heavy, i = min(cands)
active = [t for t, v in zip(tasks, y[i], strict=True) if v == 1]
print(f"{smiles[i]}\nrow {i}, {heavy} heavy atoms, active in {len(active)} of "
      f"{int((~np.isnan(y[i])).sum())} labelled assays: {', '.join(active)}")  # fmt: skip
