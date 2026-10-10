"""Central Morgan fingerprint generator using the RDKit 2024.03+ API.

`AllChem.GetMorganFingerprintAsBitVect` is deprecated in RDKit
2024.03+ and will be removed in a future release. All call sites
route through this module so the migration lives in one place.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit.Chem import rdFingerprintGenerator


@lru_cache(maxsize=16)
def _generator(radius: int, n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def morgan_fp(mol, n_bits: int, radius: int = 2):
    """Drop-in replacement for `AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)`.

    Returns an ExplicitBitVect identical in semantics to the legacy call.
    """
    return _generator(radius, n_bits).GetFingerprint(mol)
