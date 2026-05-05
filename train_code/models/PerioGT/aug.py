"""
aug.py — Polymer augmentation utilities
Adapted from PerioGT-main/PerioGT_copolym/utils/aug.py
"""
import random
from typing import List
from copy import deepcopy
from rdkit import Chem
import numpy as np


def generate_multimer_smiles(num_repeat_units, smiles, replace_dummy_atoms=True):
    """Generate multimer SMILES"""
    if num_repeat_units == 1:
        return smiles
    monomer = Chem.MolFromSmiles(smiles)
    repeat_points = []
    cnct_points = []
    bond_type = []
    num_dummy_atoms = 0
    for atom in monomer.GetAtoms():
        if atom.GetSymbol() == '*':
            repeat_points.append(atom.GetIdx())
            neis = atom.GetNeighbors()
            assert len(neis) == 1, f"*atom has more than one neighbor: {smiles}"
            cnct_points.append(atom.GetNeighbors()[0].GetIdx())
            bond_type.append(
                monomer.GetBondBetweenAtoms(atom.GetIdx(), atom.GetNeighbors()[0].GetIdx()).GetBondType())
            num_dummy_atoms += 1

    assert num_dummy_atoms == 2, "molecule has more than 2 *atoms"
    assert bond_type[0] == bond_type[1], "bond type of 2 *atoms are not same"

    num_atoms = monomer.GetNumAtoms()
    oligomer = deepcopy(monomer)
    for _ in range(num_repeat_units - 1):
        oligomer = Chem.CombineMols(oligomer, monomer)

    REPEAT_LIST, CNCT_LIST = np.zeros([num_repeat_units, 2]), np.zeros([num_repeat_units, 2])
    for i in range(num_repeat_units):
        REPEAT_LIST[i], CNCT_LIST[i] = np.array(repeat_points) + i * num_atoms, np.array(
            cnct_points) + i * num_atoms

    ed_oligomer = Chem.EditableMol(oligomer)
    removed_atoms_idx = []
    for i in range(num_repeat_units - 1):
        ed_oligomer.AddBond(int(CNCT_LIST[i, 1]), int(CNCT_LIST[i + 1, 0]), order=bond_type[0])
        removed_atoms_idx.extend([int(REPEAT_LIST[i, 1]), int(REPEAT_LIST[i + 1, 0])])

    if replace_dummy_atoms:
        ed_oligomer.ReplaceAtom(int(REPEAT_LIST[0, 0]), Chem.Atom(1))
        ed_oligomer.ReplaceAtom(int(REPEAT_LIST[num_repeat_units - 1, 1]), Chem.Atom(1))

    for i in sorted(removed_atoms_idx, reverse=True):
        ed_oligomer.RemoveAtom(i)

    final_mol = ed_oligomer.GetMol()
    final_mol = Chem.RemoveHs(final_mol)
    return Chem.MolToSmiles(final_mol)


def periodicity_augment_traverse(smiles: str) -> List[str]:
    """Traverse along the backbone to generate a list of periodic augmentation SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    atom_p = []
    atom_pn = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            atom_p.append(atom.GetIdx())
            atom_pn.append(atom.GetNeighbors()[0].GetIdx())

    p_atom_path = Chem.GetShortestPath(mol, atom_p[0], atom_p[1])
    backbone_bonds = []
    for i in range(1, len(p_atom_path) - 2):
        bond = mol.GetBondBetweenAtoms(p_atom_path[i], p_atom_path[i + 1])
        if not bond.IsInRing():
            backbone_bonds.append(bond)

    pa_samples = [smiles]
    for bond in backbone_bonds:
        begin_at_idx, end_at_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond = mol.GetBondBetweenAtoms(begin_at_idx, end_at_idx)
        bond_type = bond.GetBondType()
        ed_mol = Chem.EditableMol(mol)
        ed_mol.RemoveBond(begin_at_idx, end_at_idx)
        new_star_idx0 = ed_mol.AddAtom(Chem.Atom(0))
        new_star_idx1 = ed_mol.AddAtom(Chem.Atom(0))
        ed_mol.AddBond(new_star_idx0, begin_at_idx, order=bond_type)
        ed_mol.AddBond(new_star_idx1, end_at_idx, order=bond_type)
        ed_mol.AddBond(atom_pn[0], atom_pn[1],
                       order=mol.GetBondBetweenAtoms(atom_p[0], atom_pn[0]).GetBondType())
        ed_mol.RemoveAtom(max(atom_p))
        ed_mol.RemoveAtom(min(atom_p))
        final_mol = ed_mol.GetMol()
        pa_samples.append(Chem.MolToSmiles(final_mol))

    results = []
    [results.append(i) for i in pa_samples if i not in results]
    return results
