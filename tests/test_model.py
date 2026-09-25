import random

import torch
from rdkit import Chem
from torch_geometric.data import Batch

from molfp.featurize import ATOM_DIM, mol_to_data, smiles_to_data
from molfp.model import MolGCN

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]


def batch_of(smiles):
    return Batch.from_data_list([smiles_to_data(s) for s in smiles])


def test_forward_and_embed_shapes():
    b = batch_of(SMILES)
    for out_dim in (1, 12):
        m = MolGCN(ATOM_DIM, out_dim).eval()
        assert m(b).shape == (len(SMILES), out_dim)
        assert m.embed(b).shape == (len(SMILES), 256)


def test_permutation_invariance():
    torch.manual_seed(0)
    m = MolGCN(ATOM_DIM, 12).eval()
    rng = random.Random(0)
    for s in SMILES:
        mol = Chem.MolFromSmiles(s)
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        shuffled = Chem.RenumberAtoms(mol, order)
        a = m(Batch.from_data_list([mol_to_data(mol, s)]))
        b = m(Batch.from_data_list([mol_to_data(shuffled, s)]))
        assert (a - b).abs().max() < 1e-5


def test_forward_on_graphs_without_bonds():
    m = MolGCN(ATOM_DIM, 1).eval()
    out = m(batch_of(["C", "[Na+].[Cl-]", "CCO"]))
    assert out.shape == (3, 1) and torch.isfinite(out).all()
