import random

import torch

from molfp.featurize import ATOM_DIM, BOND_DIM, ELEMENTS, smiles_to_data


def test_benzene_graph():
    d = smiles_to_data("c1ccccc1")
    assert d.num_nodes == 6
    assert d.edge_index.shape == (2, 12)  # 6 bonds -> 12 directed edges
    assert d.num_atoms == 6
    aromatic_col = ATOM_DIM - 3  # [..., is_aromatic, is_in_ring, mass/100]
    assert torch.all(d.x[:, aromatic_col] == 1)


def test_ethanol_graph():
    d = smiles_to_data("CCO", y=-0.77)
    assert d.num_nodes == 3
    assert d.edge_index.shape == (2, 4)
    # edges are undirected: every (i, j) has a matching (j, i)
    pairs = set(map(tuple, d.edge_index.t().tolist()))
    assert all((j, i) in pairs for i, j in pairs)
    # O atom (index 2): element one-hot is the first len(ELEMENTS)+1 columns
    element = d.x[2, : len(ELEMENTS) + 1]
    assert element[ELEMENTS.index("O")] == 1 and element.sum() == 1
    assert d.smiles == "CCO"
    assert torch.allclose(d.y, torch.tensor([[-0.77]]))


def test_invalid_smiles_returns_none():
    for bad in ["C1CC", "xyz", "", "C(("]:
        assert smiles_to_data(bad) is None


def test_feature_dim_constant(need_data):
    from molfp.data import load_dataset

    smiles, _, _ = load_dataset("tox21")
    for s in random.Random(0).sample(smiles, 100):
        d = smiles_to_data(s)
        assert d is not None
        assert d.x.shape[1] == ATOM_DIM
        assert d.edge_attr.shape[0] == d.edge_index.shape[1]


def test_edge_cases():
    salt = smiles_to_data("[Na+].[Cl-]")  # two fragments, no bonds
    assert salt.num_nodes == 2 and salt.edge_index.shape == (2, 0)
    assert salt.edge_attr.shape == (0, BOND_DIM)
    single = smiles_to_data("C")
    assert single.num_nodes == 1 and single.edge_index.shape == (2, 0)
    boron = smiles_to_data("B(O)(O)O")
    assert boron.x[0, len(ELEMENTS)] == 1  # boron -> "other" element slot
    methane = smiles_to_data("C")
    h_block = methane.x[0, 21:26]  # total H count one-hot: CH4 -> "4 or more"
    assert h_block.tolist() == [0, 0, 0, 0, 1]


def test_rows_are_valid_one_hots():
    d = smiles_to_data("CC(=O)Oc1ccccc1C(=O)O")
    for start, stop in [(0, 10), (10, 17), (17, 21), (21, 26), (26, 32)]:
        assert torch.all(d.x[:, start:stop].sum(dim=1) == 1)
    assert d.x.shape[1] == 35
