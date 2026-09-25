"""SMILES -> PyTorch Geometric graph. Our own featurizer (invariant 1).

Atom features: concatenated one-hots plus two binary flags and one scalar. ATOM_DIM = 35.

| Block          | Values                                  | Width | Columns |
|----------------|-----------------------------------------|-------|---------|
| element        | C N O S F Cl Br I P, other              | 10    | 0-9     |
| degree         | 0 1 2 3 4 5, other                      | 7     | 10-16   |
| formal charge  | -1 0 +1, other                          | 4     | 17-20   |
| total H count  | 0 1 2 3 4 (4 means "4 or more")         | 5     | 21-25   |
| hybridization  | SP SP2 SP3 SP3D SP3D2, other            | 6     | 26-31   |
| is aromatic    | 0/1                                     | 1     | 32      |
| is in ring     | 0/1                                     | 1     | 33      |
| mass / 100     | float (C = 0.12)                        | 1     | 34      |

Bond features (edge_attr, BOND_DIM = 6): type one-hot (single, double, triple, aromatic),
is conjugated, is in ring. GCNConv does NOT use edge_attr; it is stored so an edge-aware
model (GAT/MPNN) could use it without touching this file. See docs/adr/001.

Graph rules (invariant 2):
- every bond i-j becomes two directed edges i->j and j->i;
- hydrogens are implicit: counted in the "total H count" block, never nodes.
"""

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")  # we report parse errors ourselves (invariant 3)

FEATURIZER_VERSION = "1"  # bump on any change; checkpoints store it

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-1, 0, 1]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

ATOM_DIM = (len(ELEMENTS) + 1) + (len(DEGREES) + 1) + (len(CHARGES) + 1) + 5
ATOM_DIM += (len(HYBRIDIZATIONS) + 1) + 3
BOND_DIM = len(BOND_TYPES) + 2


def one_hot(value, choices, other=True):
    """One-hot over `choices`; unknown values go to a final 'other' slot if other=True."""
    vec = [float(value == c) for c in choices]
    if other:
        vec.append(float(value not in choices))
    return vec


def atom_features(atom):
    return (
        one_hot(atom.GetSymbol(), ELEMENTS)
        + one_hot(atom.GetDegree(), DEGREES)
        + one_hot(atom.GetFormalCharge(), CHARGES)
        + one_hot(min(atom.GetTotalNumHs(), 4), [0, 1, 2, 3, 4], other=False)
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing()), atom.GetMass() / 100]
    )


def bond_features(bond):
    return one_hot(bond.GetBondType(), BOND_TYPES, other=False) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def parse_smiles(smiles):
    """Return (mol, None) or (None, reason). Never raises (invariant 3)."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "empty input"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, "syntax error (check brackets and ring-closure digits)"
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:  # RDKit raises several exception types for bad valence/aromaticity
        return None, f"invalid chemistry: {e}"
    if mol.GetNumAtoms() == 0:
        return None, "no atoms"
    return mol, None


def mol_to_data(mol, smiles, y=None):
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]
        dst += [j, i]
        eattr += [f, f]
    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    edge_attr = torch.tensor(eattr, dtype=torch.float).reshape(-1, BOND_DIM)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    data.num_atoms = mol.GetNumAtoms()
    if y is not None:
        # shape [1, num_tasks] so a batch stacks to [batch, num_tasks]; NaN = missing label
        data.y = torch.tensor(y, dtype=torch.float).reshape(1, -1)
    return data


def smiles_to_data(smiles, y=None):
    """SMILES -> Data(x, edge_index, edge_attr, y, smiles, num_atoms), or None if invalid."""
    mol, _ = parse_smiles(smiles)
    return None if mol is None else mol_to_data(mol, smiles, y)
