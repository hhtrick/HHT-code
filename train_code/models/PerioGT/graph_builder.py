"""
graph_builder.py — PerioGT Graph Construction (PyTorch Geometric version)
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data
from functools import partial
from itertools import permutations
import networkx as nx

from .constants import (
    VIRTUAL_PATH_INDICATOR, VIRTUAL_ATOM_FEATURE_PLACEHOLDER,
    VIRTUAL_BOND_FEATURE_PLACEHOLDER, VIRTUAL_ATOM_INDICATOR
)

D_ATOM_FEATS = 138
D_BOND_FEATS = 14

# ---------------------------------------------------------------------------
# Pure RDKit featurizer (equivalent to dgllife version, exactly matching output dimensions)
# When encode_unknown=True, appends a one-dimensional unknown flag at the end, making output length
# len(allowable_set) + 1 (matching dgllife behavior).
# ---------------------------------------------------------------------------

def _one_hot_enc(x, allowable_set, encode_unknown=False):
    """Equivalent implementation of dgllife's one_hot_encoding."""
    if encode_unknown:
        if x not in allowable_set:
            return [0] * len(allowable_set) + [1]
        return [int(x == s) for s in allowable_set] + [0]
    return [int(x == s) for s in allowable_set]


# ---- Atom features ----

def _atomic_number_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(1, 119))
    return _one_hot_enc(atom.GetAtomicNum(), allowable_set, encode_unknown)


def _atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(0, 11))
    return _one_hot_enc(atom.GetDegree(), allowable_set, encode_unknown)


def _atom_formal_charge(atom):
    return [atom.GetFormalCharge()]


def _atom_num_radical_electrons_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(0, 5))
    return _one_hot_enc(atom.GetNumRadicalElectrons(), allowable_set, encode_unknown)


def _atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [
            rdchem.HybridizationType.SP,
            rdchem.HybridizationType.SP2,
            rdchem.HybridizationType.SP3,
            rdchem.HybridizationType.SP3D,
            rdchem.HybridizationType.SP3D2,
        ]
    return _one_hot_enc(atom.GetHybridization(), allowable_set, encode_unknown)


def _atom_is_aromatic(atom):
    return [int(atom.GetIsAromatic())]


def _atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(0, 5))
    return _one_hot_enc(atom.GetTotalNumHs(), allowable_set, encode_unknown)


def _atom_is_chiral_center(atom):
    return [int(atom.HasProp('_CIPCode'))]


def _atom_chirality_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = ['R', 'S']
    try:
        chirality = atom.GetPropsAsDict().get('_CIPCode', '')
    except Exception:
        chirality = ''
    return _one_hot_enc(chirality, allowable_set, encode_unknown)


def _atom_mass(atom, coef=0.01):
    return [atom.GetMass() * coef]


# ---- Bond features ----

def _bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [
            rdchem.BondType.SINGLE,
            rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE,
            rdchem.BondType.AROMATIC,
        ]
    return _one_hot_enc(bond.GetBondType(), allowable_set, encode_unknown)


def _bond_is_conjugated(bond):
    return [int(bond.GetIsConjugated())]


def _bond_is_in_ring(bond):
    return [int(bond.IsInRing())]


def _bond_stereo_one_hot(bond, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [
            rdchem.BondStereo.STEREONONE,
            rdchem.BondStereo.STEREOANY,
            rdchem.BondStereo.STEREOZ,
            rdchem.BondStereo.STEREOE,
            rdchem.BondStereo.STEREOCIS,
            rdchem.BondStereo.STEREOTRANS,
        ]
    return _one_hot_enc(bond.GetStereo(), allowable_set, encode_unknown)


# ---- ConcatFeaturizer ----

class _ConcatFeaturizer:
    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, x):
        result = []
        for f in self.funcs:
            result.extend(f(x))
        return result


def _build_atom_featurizer():
    # Output dimension: 102+12+1+6+6+1+6+1+2+1 = 138
    return _ConcatFeaturizer([
        partial(_atomic_number_one_hot, allowable_set=list(range(0, 101)), encode_unknown=True),  # 102
        partial(_atom_degree_one_hot, encode_unknown=True),                                        # 12
        _atom_formal_charge,                                                                        # 1
        partial(_atom_num_radical_electrons_one_hot, encode_unknown=True),                         # 6
        partial(_atom_hybridization_one_hot, encode_unknown=True),                                 # 6
        _atom_is_aromatic,                                                                          # 1
        partial(_atom_total_num_H_one_hot, encode_unknown=True),                                   # 6
        _atom_is_chiral_center,                                                                     # 1
        _atom_chirality_type_one_hot,                                                               # 2
        _atom_mass,                                                                                 # 1
    ])


def _build_bond_featurizer():
    # Output dimension: 5+1+1+7 = 14
    return _ConcatFeaturizer([
        partial(_bond_type_one_hot, encode_unknown=True),    # 5
        _bond_is_conjugated,                                  # 1
        _bond_is_in_ring,                                     # 1
        partial(_bond_stereo_one_hot, encode_unknown=True),  # 7
    ])



def preprocess_batch_light(batch_num, batch_num_target, tensor_data):
    """Preprocess path index offsets for each graph in the batch"""
    batch_num = np.concatenate([[0], batch_num], axis=-1)
    cs_num = np.cumsum(batch_num)
    add_factors = np.concatenate(
        [[cs_num[i]] * batch_num_target[i] for i in range(len(cs_num) - 1)], axis=-1
    )
    return tensor_data + torch.from_numpy(add_factors).reshape(-1, 1)


class PolyGraphBuilder:
    """
    Polymer graph builder.
    Constructs a PyG Data graph with local virtual nodes (per component) and global virtual nodes.
    Supports 1 or 2 components (homopolymer/copolymer).
    """
    GRAPH_KEYS = [
        'edges', 'atom_pairs_features_in_triplets', 'bond_features_in_triplets',
        'virtual_atom_and_virtual_node_labels', 'paths',
        'line_graph_path_labels', 'mol_graph_path_labels',
        'virtual_path_labels', 'self_loop_labels'
    ]

    def __init__(self, max_length=5, n_local_nodes=3, n_global_nodes=1, add_self_loop=True):
        """
        Args:
            n_local_nodes: number of local virtual nodes per component (default 3: fp, md, ratio)
            n_global_nodes: number of global virtual nodes (default 1: copolymer type)
        """
        self.atom_featurizer = _build_atom_featurizer()
        self.bond_featurizer = _build_bond_featurizer()
        self.max_length = max_length
        self.n_local_nodes = n_local_nodes
        self.n_global_nodes = n_global_nodes
        self.add_self_loop = add_self_loop

    def build_copolymer_graph(self, smiles_list):
        """
        Build a copolymer graph.
        Args:
            smiles_list: list of SMILES strings (1-2 components)
        Returns:
            torch_geometric.data.Data graph
        """
        polygraph_data = {k: [] for k in self.GRAPH_KEYS}
        smiles_list = list(filter(None, smiles_list))

        node_offset = 0
        for i, smiles in enumerate(smiles_list):
            result = self._build_subgraph(
                smiles=smiles, max_length=self.max_length,
                n_local_nodes=self.n_local_nodes,
                add_self_loop=self.add_self_loop,
                node_offset=node_offset, subgraph_idx=i
            )
            if result is None:
                continue
            self._extend_graph_data(polygraph_data, result)
            node_offset = max([v for edge in polygraph_data['edges'] for v in edge]) + 1

        # Add global virtual nodes
        for n in range(self.n_global_nodes):
            current_node_idx = len(polygraph_data['atom_pairs_features_in_triplets'])
            for i in range(current_node_idx):
                if polygraph_data['virtual_atom_and_virtual_node_labels'][i] > 0:
                    continue
                polygraph_data['edges'].append([current_node_idx, i])
                polygraph_data['edges'].append([i, current_node_idx])
                polygraph_data['paths'].append(
                    [current_node_idx] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [i])
                polygraph_data['paths'].append(
                    [i] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [current_node_idx])
                polygraph_data['line_graph_path_labels'].extend([0, 0])
                polygraph_data['mol_graph_path_labels'].extend([0, 0])
                vp_label = self.n_local_nodes * len(smiles_list) + n + 1
                polygraph_data['virtual_path_labels'].extend([vp_label, vp_label])
                polygraph_data['self_loop_labels'].extend([0, 0])
            polygraph_data['atom_pairs_features_in_triplets'].append(
                [[VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * D_ATOM_FEATS,
                 [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * D_ATOM_FEATS])
            polygraph_data['bond_features_in_triplets'].append(
                [VIRTUAL_BOND_FEATURE_PLACEHOLDER] * D_BOND_FEATS)
            polygraph_data['virtual_atom_and_virtual_node_labels'].append(
                self.n_local_nodes * len(smiles_list) + n + 1)

        edges_list = polygraph_data['edges']
        N = len(polygraph_data['atom_pairs_features_in_triplets'])
        if edges_list:
            edges = np.array(edges_list, dtype=np.int64).reshape(-1, 2)
            edge_index = torch.tensor(edges.T, dtype=torch.long).contiguous()  # [2, E]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(
            edge_index=edge_index,
            num_nodes=N,
            begin_end=torch.FloatTensor(polygraph_data['atom_pairs_features_in_triplets']),
            edge=torch.FloatTensor(polygraph_data['bond_features_in_triplets']),
            vavn=torch.LongTensor(polygraph_data['virtual_atom_and_virtual_node_labels']),
            path=torch.LongTensor(polygraph_data['paths']),
            lgp=torch.BoolTensor(polygraph_data['line_graph_path_labels']),
            mgp=torch.BoolTensor(polygraph_data['mol_graph_path_labels']),
            vp=torch.BoolTensor(polygraph_data['virtual_path_labels']),
            sl=torch.BoolTensor(polygraph_data['self_loop_labels']),
        )
        return data

    def _build_subgraph(self, smiles, max_length=5, n_local_nodes=3,
                        node_offset=0, subgraph_idx=0, add_self_loop=True):
        """Build a sub-graph for a single component"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            import warnings
            warnings.warn(f"[PerioGT] Invalid SMILES skipped in _build_subgraph: '{smiles}'")
            return None
        new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
        mol = Chem.rdmolops.RenumberAtoms(mol, new_order)

        n_atoms = mol.GetNumAtoms()
        atom_features = []
        for atom_id in range(n_atoms):
            atom = mol.GetAtomWithIdx(atom_id)
            atom_features.append(self.atom_featurizer(atom))

        atomIDPair_to_tripletId = np.ones(shape=(n_atoms, n_atoms)) * np.nan

        virtual_atom_and_virtual_node_labels = []
        atom_pairs_features_in_triplets = []
        bond_features_in_triplets = []
        bonded_atoms = set()
        triplet_id = node_offset

        bonds = sorted(mol.GetBonds(), key=lambda b: b.GetIdx())
        for bond in bonds:
            begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
            atom_pairs_features_in_triplets.append([atom_features[begin_atom_id], atom_features[end_atom_id]])
            bond_features_in_triplets.append(self.bond_featurizer(bond))
            bonded_atoms.add(begin_atom_id)
            bonded_atoms.add(end_atom_id)
            virtual_atom_and_virtual_node_labels.append(0)
            atomIDPair_to_tripletId[begin_atom_id, end_atom_id] = triplet_id
            atomIDPair_to_tripletId[end_atom_id, begin_atom_id] = triplet_id
            triplet_id += 1

        for atom_id in range(n_atoms):
            if atom_id not in bonded_atoms:
                atom_pairs_features_in_triplets.append(
                    [atom_features[atom_id], [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * D_ATOM_FEATS])
                bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * D_BOND_FEATS)
                virtual_atom_and_virtual_node_labels.append(VIRTUAL_ATOM_INDICATOR)

        edges, paths = [], []
        line_graph_path_labels, mol_graph_path_labels = [], []
        virtual_path_labels, self_loop_labels = [], []

        for i in range(n_atoms):
            node_ids = atomIDPair_to_tripletId[i]
            node_ids = node_ids[~np.isnan(node_ids)]
            if len(node_ids) >= 2:
                new_edges = list(permutations(node_ids, 2))
                edges.extend(new_edges)
                new_paths = [[e[0]] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [e[1]] for e in new_edges]
                paths.extend(new_paths)
                n_new = len(new_edges)
                line_graph_path_labels.extend([1] * n_new)
                mol_graph_path_labels.extend([0] * n_new)
                virtual_path_labels.extend([0] * n_new)
                self_loop_labels.extend([0] * n_new)

        adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
        nx_g = nx.from_numpy_array(adj_matrix)
        paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_g, max_length + 1))
        for i in paths_dict.keys():
            for j in paths_dict[i]:
                path = paths_dict[i][j]
                path_length = len(path)
                if 3 < path_length <= max_length + 1:
                    triplet_ids = [atomIDPair_to_tripletId[path[pi], path[pi + 1]]
                                   for pi in range(len(path) - 1)]
                    triplet_path = [triplet_ids[0]] + list(triplet_ids[1:-1]) + \
                                   [VIRTUAL_PATH_INDICATOR] * (max_length - len(triplet_ids[1:-1]) - 2) + \
                                   [triplet_ids[-1]]
                    paths.append(triplet_path)
                    edges.append([triplet_ids[0], triplet_ids[-1]])
                    line_graph_path_labels.append(0)
                    mol_graph_path_labels.append(1)
                    virtual_path_labels.append(0)
                    self_loop_labels.append(0)

        # Add local virtual nodes
        for j, n in enumerate(range(subgraph_idx * n_local_nodes, (subgraph_idx + 1) * n_local_nodes)):
            for i in range(len(atom_pairs_features_in_triplets) - j):
                edges.append([len(atom_pairs_features_in_triplets) + node_offset, i + node_offset])
                edges.append([i + node_offset, len(atom_pairs_features_in_triplets) + node_offset])
                paths.append([len(atom_pairs_features_in_triplets) + node_offset] +
                             [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [i + node_offset])
                paths.append([i + node_offset] +
                             [VIRTUAL_PATH_INDICATOR] * (max_length - 2) +
                             [len(atom_pairs_features_in_triplets) + node_offset])
                line_graph_path_labels.extend([0, 0])
                mol_graph_path_labels.extend([0, 0])
                virtual_path_labels.extend([n + 1, n + 1])
                self_loop_labels.extend([0, 0])
            atom_pairs_features_in_triplets.append(
                [[VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * D_ATOM_FEATS,
                 [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * D_ATOM_FEATS])
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * D_BOND_FEATS)
            virtual_atom_and_virtual_node_labels.append(n + 1)

        if add_self_loop:
            for i in range(len(atom_pairs_features_in_triplets)):
                edges.append([i + node_offset, i + node_offset])
                paths.append([i + node_offset] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [i + node_offset])
                line_graph_path_labels.append(0)
                mol_graph_path_labels.append(0)
                virtual_path_labels.append(0)
                self_loop_labels.append(1)

        return (edges, atom_pairs_features_in_triplets, bond_features_in_triplets,
                virtual_atom_and_virtual_node_labels, paths,
                line_graph_path_labels, mol_graph_path_labels,
                virtual_path_labels, self_loop_labels)

    def _extend_graph_data(self, acc, new_data):
        for k, v in zip(self.GRAPH_KEYS, new_data):
            acc[k].extend(v)

    def build_prompt_graph(self, smiles: str):
        """
        Build a simple single-component graph (n_local=2, n_global=0) for prompt generation, with bid annotations added.
        bid (bond identity) is used to map enhanced graph node embeddings back to the base graph.
        bonded triplet: bid = canonical bond index; virtual node: bid = -1.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            import warnings
            warnings.warn(f"[PerioGT] Invalid SMILES skipped in build_prompt_graph: '{smiles}'")
            return None
        new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
        mol = Chem.rdmolops.RenumberAtoms(mol, new_order)

        n_atoms = mol.GetNumAtoms()
        atom_features = []
        for atom_id in range(n_atoms):
            atom = mol.GetAtomWithIdx(atom_id)
            atom_features.append(self.atom_featurizer(atom))

        atomIDPair_to_tripletId = np.ones(shape=(n_atoms, n_atoms)) * np.nan

        virtual_atom_and_virtual_node_labels = []
        atom_pairs_features_in_triplets = []
        bond_features_in_triplets = []
        bid_list = []  # bond identity
        bonded_atoms = set()
        triplet_id = 0

        bonds = sorted(mol.GetBonds(), key=lambda b: b.GetIdx())
        for bond_idx, bond in enumerate(bonds):
            begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
            atom_pairs_features_in_triplets.append([atom_features[begin_atom_id], atom_features[end_atom_id]])
            bond_features_in_triplets.append(self.bond_featurizer(bond))
            bonded_atoms.add(begin_atom_id)
            bonded_atoms.add(end_atom_id)
            virtual_atom_and_virtual_node_labels.append(0)
            bid_list.append(bond_idx)
            atomIDPair_to_tripletId[begin_atom_id, end_atom_id] = triplet_id
            atomIDPair_to_tripletId[end_atom_id, begin_atom_id] = triplet_id
            triplet_id += 1

        for atom_id in range(n_atoms):
            if atom_id not in bonded_atoms:
                atom_pairs_features_in_triplets.append(
                    [atom_features[atom_id], [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * D_ATOM_FEATS])
                bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * D_BOND_FEATS)
                virtual_atom_and_virtual_node_labels.append(VIRTUAL_ATOM_INDICATOR)
                bid_list.append(-1)

        edges, paths = [], []
        line_graph_path_labels, mol_graph_path_labels = [], []
        virtual_path_labels, self_loop_labels = [], []

        # line graph paths
        for i in range(n_atoms):
            node_ids = atomIDPair_to_tripletId[i]
            node_ids = node_ids[~np.isnan(node_ids)]
            if len(node_ids) >= 2:
                new_edges = list(permutations(node_ids, 2))
                edges.extend(new_edges)
                new_paths = [[e[0]] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [e[1]] for e in new_edges]
                paths.extend(new_paths)
                n_new = len(new_edges)
                line_graph_path_labels.extend([1] * n_new)
                mol_graph_path_labels.extend([0] * n_new)
                virtual_path_labels.extend([0] * n_new)
                self_loop_labels.extend([0] * n_new)

        # molecule graph paths
        adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
        nx_g = nx.from_numpy_array(adj_matrix)
        paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_g, self.max_length + 1))
        for i in paths_dict.keys():
            for j in paths_dict[i]:
                path = paths_dict[i][j]
                path_length = len(path)
                if 3 < path_length <= self.max_length + 1:
                    triplet_ids = [atomIDPair_to_tripletId[path[pi], path[pi + 1]]
                                   for pi in range(len(path) - 1)]
                    triplet_path = [triplet_ids[0]] + list(triplet_ids[1:-1]) + \
                                   [VIRTUAL_PATH_INDICATOR] * (self.max_length - len(triplet_ids[1:-1]) - 2) + \
                                   [triplet_ids[-1]]
                    paths.append(triplet_path)
                    edges.append([triplet_ids[0], triplet_ids[-1]])
                    line_graph_path_labels.append(0)
                    mol_graph_path_labels.append(1)
                    virtual_path_labels.append(0)
                    self_loop_labels.append(0)

        # Add 2 local virtual nodes (fp, md)
        n_local = 2
        for j in range(n_local):
            n_existing = len(atom_pairs_features_in_triplets)
            for i in range(n_existing - j):
                edges.append([n_existing, i])
                edges.append([i, n_existing])
                paths.append([n_existing] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [i])
                paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [n_existing])
                line_graph_path_labels.extend([0, 0])
                mol_graph_path_labels.extend([0, 0])
                virtual_path_labels.extend([j + 1, j + 1])
                self_loop_labels.extend([0, 0])
            atom_pairs_features_in_triplets.append(
                [[VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * D_ATOM_FEATS,
                 [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * D_ATOM_FEATS])
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * D_BOND_FEATS)
            virtual_atom_and_virtual_node_labels.append(j + 1)
            bid_list.append(-1)

        # Self-loop
        if self.add_self_loop:
            for i in range(len(atom_pairs_features_in_triplets)):
                edges.append([i, i])
                paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [i])
                line_graph_path_labels.append(0)
                mol_graph_path_labels.append(0)
                virtual_path_labels.append(0)
                self_loop_labels.append(1)

        if not edges:
            return None

        N = len(atom_pairs_features_in_triplets)
        edges = np.array(edges, dtype=np.int64)
        edge_index = torch.tensor(edges.T, dtype=torch.long).contiguous()  # [2, E]

        data = Data(
            edge_index=edge_index,
            num_nodes=N,
            begin_end=torch.FloatTensor(atom_pairs_features_in_triplets),
            edge=torch.FloatTensor(bond_features_in_triplets),
            vavn=torch.LongTensor(virtual_atom_and_virtual_node_labels),
            bid=torch.LongTensor(bid_list),
            node_mask=torch.zeros(N, dtype=torch.long),
            path=torch.LongTensor(paths),
            lgp=torch.BoolTensor(line_graph_path_labels),
            mgp=torch.BoolTensor(mol_graph_path_labels),
            vp=torch.BoolTensor(virtual_path_labels),
            sl=torch.BoolTensor(self_loop_labels),
        )
        return data
