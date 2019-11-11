"""Molecule Feature Description
Unless otherwise noted, all work by:
******************************************************************
Title: PA-Graph-Transformer
Author: Benson Chen (benatorc@gmail.com)
Date: May 28, 2019
Code version: 4274301
Availability: https://github.com/benatorc/PA-Graph-Transformer.git
******************************************************************
"""
import rdkit.Chem as Chem
import torch as torch

# The default valid symbols for atom features
SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
           'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
           'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
           'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
           'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re',
           'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm',
           'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', '*', 'UNK']

# The default valid formal charges for atom features
FORMAL_CHARGES = [-2, -1, 0, 1, 2]

CHIRAL_TAG = [0,1,2,3]

# The default valid bond types for bond features
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    None,  # Zero, no bond
]

BT_MAPPING = {
    0.: None,
    1.: Chem.rdchem.BondType.SINGLE,
    2.: Chem.rdchem.BondType.DOUBLE,
    3.: Chem.rdchem.BondType.TRIPLE,
    1.5: Chem.rdchem.BondType.AROMATIC,
}

BT_MAPPING_INV = {v: k for k, v in BT_MAPPING.items()}

BT_STEREO = [0,1,2,3,4,5]

def bt_index_to_float(bt_index):
    bond_type = BOND_TYPES[bt_index]
    return BT_MAPPING_INV[bond_type]


# Maximum number of neighbors for an atom
MAX_NEIGHBORS = 10
DEGREES = list(range(MAX_NEIGHBORS))

EXPLICIT_VALENCES = [0, 1, 2, 3, 4, 5, 6]
IMPLICIT_VALENCES = [0, 1, 2, 3, 4, 5]

N_ATOM_FEATS = (len(SYMBOLS) + len(FORMAL_CHARGES) + len(DEGREES) +
                len(EXPLICIT_VALENCES) + len(IMPLICIT_VALENCES) + 
                len(CHIRAL_TAG) + 1)
N_BOND_FEATS = len(BOND_TYPES) + len(BT_STEREO) + 1 + 1


def get_bt_index(bond_type):
    """Returns the feature index for a particular bond type.

    Args:
        bond_type: Either a rdchem bond type object (can be None) or a float
            representing the bond type
    """
    if bond_type not in BOND_TYPES:
        assert bond_type in BT_MAPPING
        bond_type = BT_MAPPING[bond_type]

    return BOND_TYPES.index(bond_type)


def onek_unk_encoding(x, set):
    """Returns a one-hot encoding of the given feature."""
    if x not in set:
        x = 'UNK'
    return [int(x == s) for s in set]


def get_atom_features(atom):
    """Given an atom object, returns a numpy array of features."""
    # Atom features are symbol, formal charge, degree, explicit/implicit
    # valence, and aromaticity
    symbol = onek_unk_encoding(atom.GetSymbol(), SYMBOLS)

    if False:  # atom.is_dummy:
        padding = [0] * (N_ATOM_FEATS - len(symbol))
        feature_array = symbol + padding
    else:
        aro = [atom.GetIsAromatic()]
        chiral = onek_unk_encoding(int(atom.GetChiralTag()), CHIRAL_TAG)
        degree = onek_unk_encoding(atom.GetDegree(), DEGREES)
        exp_valence = onek_unk_encoding(atom.GetExplicitValence(), 
                                        EXPLICIT_VALENCES)        
        fc = onek_unk_encoding(atom.GetFormalCharge(), FORMAL_CHARGES)
        imp_valence = onek_unk_encoding(atom.GetImplicitValence(), 
                                        IMPLICIT_VALENCES)

        feature_array = symbol + aro + chiral + degree + exp_valence + \
                        fc + imp_valence
    return torch.Tensor(feature_array)


def get_bond_features(bond, bt_only=False):
    """Given an bond object, returns a numpy array of features.

    bond can be None, in which case returns default features for a non-bond.
    """
    # Bond features are bond type, conjugacy, and ring-membership
    if bond is None:
        bond_type = onek_unk_encoding(None, BOND_TYPES)
        stereo = [0]
        conj = [0]
        ring = [0]
    else:
        bond_type = onek_unk_encoding(bond.GetBondType(), BOND_TYPES)
        stereo = [int(bond.GetStereo())]
        fstereo = onek_unk_encoding(stereo, BT_STEREO)
        conj = [bond.GetIsConjugated()]
        ring = [bond.IsInRing()]

    if bt_only:
        feature_array = bond_type
    else:   
        feature_array = bond_type + fstereo + conj + ring
    return torch.Tensor(feature_array)


def get_bt_feature(bond_type):
    """Returns a one-hot vector representing the bond_type."""
    if bond_type in BT_MAPPING:
        bond_type = BT_MAPPING[bond_type]
    return onek_unk_encoding(bond_type, BOND_TYPES)


def get_path_bond_feature(bond):
    """Given a rdkit bond object, returns the bond features for that bond.

    When the given input is none, returns a 0-vector"""
    if bond is None:
        return torch.zeros(N_BOND_FEATS)
    else:
        bond_type = onek_unk_encoding(bond.GetBondType(), BOND_TYPES)
        conj = [int(bond.GetIsConjugated())]
        ring = [int(bond.IsInRing())]

        return torch.Tensor(bond_type + conj + ring)