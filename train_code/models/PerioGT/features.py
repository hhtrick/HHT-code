"""
features.py — PerioGT molecular fingerprint and descriptor computation
Adapted from PerioGT-main/PerioGT_copolym/utils/features.py
"""
import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from mordred import Calculator, descriptors
from mordred.MolecularDistanceEdge import MolecularDistanceEdge
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from .aug import generate_multimer_smiles


class _StableMolecularDistanceEdge(MolecularDistanceEdge):
    """Mordred's MDE definition without overflowing the intermediate product."""

    __slots__ = ()

    def calculate(self, D, V):
        atomic_numbers = [a.GetAtomicNum() for a in self.mol.GetAtoms()]
        distances = [
            D[i, j]
            for i in range(len(atomic_numbers))
            for j in range(i + 1, len(atomic_numbers))
            if ((V[i] == self._valence1 and V[j] == self._valence2)
                or (V[j] == self._valence1 and V[i] == self._valence2))
            and atomic_numbers[i] == atomic_numbers[j] == self._atomic_num
        ]
        # Original: dx = prod(distances)**(1/(2*n)); MDE = n/dx**2.
        # n / exp(mean(log(distances))) is equivalent, without the huge product.
        # Keep Mordred's missing-value behavior when no qualifying pair exists.
        with self.rethrow_zerodiv():
            mean_log_distance = float(np.log(distances).sum()) / len(distances)
            return float(len(distances) / np.exp(mean_log_distance))


_GLOBAL_CALC = Calculator(descriptors, ignore_3D=True)
# Replace only the affected descriptors in place; retain all 1613 names/columns.
_GLOBAL_CALC.descriptors = [
    _StableMolecularDistanceEdge(*d.parameters())
    if isinstance(d, MolecularDistanceEdge) else d
    for d in _GLOBAL_CALC.descriptors
]
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024)


def safe_mol_from_smiles(smiles: str):
    """Parse SMILES quietly; return None for invalid or unsupported structures."""
    if not smiles:
        return None

    blocker = rdBase.BlockLogs()
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    finally:
        del blocker


def _bitvect_to_array(bitvect) -> np.ndarray:
    arr = np.zeros((bitvect.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr.astype(np.float32)


def _fp_md_from_smiles(smiles: str):
    """Compute MACCS+ECFP fingerprints and Mordred descriptors for a single SMILES"""
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return smiles, None, None

    maccs_fp = _bitvect_to_array(MACCSkeys.GenMACCSKeys(mol))
    ec_fp = _bitvect_to_array(_MORGAN_GENERATOR.GetFingerprint(mol))
    fp = np.concatenate([maccs_fp, ec_fp], axis=0)

    # Use float64 first to clean extreme values, then downcast to float32.
    # Retain the existing missing/extreme-value policy for model inputs.
    des = np.array(list(_GLOBAL_CALC(mol).values()), dtype=np.float64)
    des = np.nan_to_num(des, nan=0.0, posinf=1e6, neginf=-1e6)
    des = np.clip(des, -1e6, 1e6).astype(np.float32)
    return smiles, fp, des


def compute_single_smiles_features(smiles: str):
    """
    Compute fingerprints and descriptors for a single SMILES.
    Returns:
        (fp, md) — numpy arrays; (None, None) if SMILES is invalid
    """
    _, fp, md = _fp_md_from_smiles(smiles)
    if fp is None:
        return None, None
    return np.asarray(fp, dtype=np.float32), np.asarray(md, dtype=np.float32)


def _gen_multimer_smiles(base: str, units: int):
    try:
        return generate_multimer_smiles(num_repeat_units=units, smiles=base)
    except Exception:
        return None


def precompute_features(base_smiles_list, units=(3, 6, 9), workers=None):
    """
    Precompute molecular fingerprint and descriptor cache.
    Args:
        base_smiles_list: list of base SMILES
        units: multimer repeat unit count
        workers: number of parallel processes
    Returns:
        feat_cache: dict {(base_smiles, unit): (fp, md)}
    """
    workers = workers or max(1, cpu_count() - 1)
    tasks = [(s, u) for s in base_smiles_list for u in units]
    with Pool(processes=workers) as pool:
        gen_smiles = pool.starmap(_gen_multimer_smiles, tasks, chunksize=64)

    oligos = [(base, u, sm) for (base, u), sm in zip(tasks, gen_smiles)]
    oligo_smiles_map = {}
    uniq_smiles_set = set()
    for base, u, sm in oligos:
        if sm is not None:
            oligo_smiles_map[(base, u)] = sm
            uniq_smiles_set.add(sm)

    uniq_smiles = list(uniq_smiles_set)
    if not uniq_smiles:
        print("    No multimers generated (two compatible '*' connection points are required); "
              "single-molecule features will still be computed.")
        return {}
    with Pool(processes=workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_fp_md_from_smiles, uniq_smiles, chunksize=1),
            total=len(uniq_smiles), ncols=100, desc="Multimer features",
        ))

    fpmd_map = {sm: (fp, md) for sm, fp, md in results if fp is not None}
    feat_cache = {}
    for k, sm in oligo_smiles_map.items():
        feat_cache[k] = fpmd_map.get(sm, (None, None))

    return feat_cache
