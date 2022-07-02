"""
Chemistry Utilities
===================

Chemistry and molecule utility functions.
"""
import numpy as np
import bisect
from rdkit.Chem import TorsionFingerprints
import rdkit.Chem.AllChem as Chem

from typing import Tuple, List
import logging

def prune_last_conformer(mol: Chem.Mol, tfd_thresh: float, energies: List[float]) -> Tuple[Chem.Mol, List[float]]:
    """Prunes the last conformer of the molecule.

    If no conformers in `mol` have a TFD (Torsional Fingerprint Deviation) with the last conformer of less than `tfd_thresh`,
    the last conformer is kept. Otherwise, the lowest energy conformer with TFD less than `tfd_thresh` is kept and all other conformers
    are discarded.

    Parameters
    ----------
    mol : RDKit Mol
        The molecule to be pruned. The conformers in the molecule should be ordered by ascending energy.
    tfd_thresh : float
        The minimum threshold for TFD between conformers.
    energies : list of float
        A list of all the energies of the conformers in `mol`.

    Returns
    -------
    mol : RDKit Mol
        The updated molecule after pruning, with conformers sorted by ascending energy.
    energies : list of float
        A list of all the energies of the conformers in `mol` after pruning and sorting by ascending energy.
    """
    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol, energies

    idx = bisect.bisect(energies[:-1], energies[-1])
    tfd = TorsionFingerprints.GetTFDBetweenConformers(mol, range(0, mol.GetNumConformers() - 1), [mol.GetNumConformers() - 1], useWeights=False)
    tfd = np.array(tfd)

    # if lower energy conformer is within threshold, drop new conf
    if not np.all(tfd[:idx] >= tfd_thresh):
        energies = energies[:-1]
        mol.RemoveConformer(mol.GetNumConformers() - 1)
        return mol, energies
    else:
        keep = list(range(0, idx))
        keep.append(mol.GetNumConformers() - 1)
        keep += [x for x in range(idx, mol.GetNumConformers() - 1) if tfd[x] >= tfd_thresh]

        new = Chem.Mol(mol)
        new.RemoveAllConformers()
        for i in keep:
            conf = mol.GetConformer(i)
            new.AddConformer(conf, assignId=True)

        return new, [energies[i] for i in keep]

def tfd_matrix(mol: Chem.Mol) -> np.array:
    """Calculates the TFD matrix for all conformers in a molecule.
    """
    tfd = TorsionFingerprints.GetTFDMatrix(mol, useWeights=False)
    n = int(np.sqrt(len(tfd)*2))+1
    idx = np.tril_indices(n, k=-1, m=n)
    matrix = np.zeros((n,n))
    matrix[idx] = tfd
    matrix += np.transpose(matrix)
    return matrix
