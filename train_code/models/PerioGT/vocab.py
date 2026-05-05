"""
vocab.py — PerioGT 词汇表（来自 PerioGT-main/PerioGT_copolym/data/vocab.py）
"""
import numpy as np


class Vocab(object):
    def __init__(self, n_atom_types=102, n_bond_types=5):
        self.n_atom_types = n_atom_types
        self.n_bond_types = n_bond_types
        self.vocab = self.construct()

    def construct(self):
        vocab = {}
        atom_ids = list(range(self.n_atom_types))
        bond_ids = list(range(self.n_bond_types))
        id = 0
        for atom_id_1 in atom_ids:
            vocab[atom_id_1] = {}
            for bond_id in bond_ids:
                vocab[atom_id_1][bond_id] = {}
                for atom_id_2 in atom_ids:
                    if atom_id_2 >= atom_id_1:
                        vocab[atom_id_1][bond_id][atom_id_2] = id
                        id += 1
        for atom_id in atom_ids:
            vocab[atom_id][999] = {}
            vocab[atom_id][999][999] = id
            id += 1
        vocab[999] = {}
        vocab[999][999] = {}
        vocab[999][999][999] = id
        self.vocab_size = id
        return vocab

    def __getitem__(self, items):
        atom_id_1, bond_id, atom_id_2 = items
        atom_id_1, atom_id_2 = np.sort([atom_id_1, atom_id_2])
        return self.vocab[atom_id_1][bond_id][atom_id_2]

    def index(self, atom_type1, atom_type2, bond_type):
        atom_type1, atom_type2 = np.sort([atom_type1, atom_type2])
        try:
            return self.vocab[atom_type1][bond_type][atom_type2]
        except Exception:
            return self.vocab_size
