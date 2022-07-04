"""
ForceField Mixins
===================

Different FF calculation/pruning behaviors based on the FF/simulator being used.
"""
from idp_rl.utils import chem_utils

import torch
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u
import rdkit.Chem.AllChem as Chem
from rdkit.Geometry import Point3D

from typing import Tuple, List, Callable

def np_to_mm(arr: np.ndarray, unit: openmm.unit=u.angstrom):
    wrapped_val = openmm.unit.quantity.Quantity(arr, unit)
    return wrapped_val

def calculate_normalizers(mol: Chem.Mol, optimize_conf : Callable, get_conformer_energies : Callable, num_confs: int = 200, pruning_thresh: float = 0.05) -> Tuple[float, float]:
    """Calculates the :math:`E_0` and :math:`Z_0` normalizing constants for a molecule used in the TorsionNet [1]_ paper.

    Parameters
    ----------
    mol : RDKit Mol
        The molecule of interest.
    num_confs : int
        The number of conformers to generate when calculating the constants. Should equal
        the number of steps for each episode of the environment containing this molecule.
    pruning_thresh : float
        TFD threshold for pruning the conformers of `mol`.

    References
    ----------
    .. [1] `TorsionNet paper <https://arxiv.org/abs/2006.07078>`_
    """
    Chem.MMFFSanitizeMolecule(mol)
    confslist = Chem.EmbedMultipleConfs(mol, numConfs = num_confs, useRandomCoords=True)
    if (len(confslist) < 1):
        raise Exception('Unable to embed molecule with conformer using rdkit')
    
    for conf_id in range(mol.GetNumConformers()):
        optimize_conf(mol, conf_id)
    mol = prune_conformers(mol, get_conformer_energies, pruning_thresh)
    energys = get_conformer_energies(mol)
    E0 = energys.min()
    Z0 = np.sum(np.exp(-(energys - E0)))

    mol.RemoveAllConformers()

    return E0, Z0

def prune_conformers(mol: Chem.Mol, get_conformer_energies : Callable, tfd_thresh: float) -> Chem.Mol:
    """Prunes all the conformers in the molecule.

    Removes conformers that have a TFD (torsional fingerprint deviation) lower than
    `tfd_thresh` with other conformers. Lowest energy conformers are kept.

    Parameters
    ----------
    mol : RDKit Mol
        The molecule to be pruned.
    tfd_thresh : float
        The minimum threshold for TFD between conformers.

    Returns
    -------
    mol : RDKit Mol
        The updated molecule after pruning.
    """
    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol

    energies = get_conformer_energies(mol)
    tfd = chem_utils.tfd_matrix(mol)
    sort = np.argsort(energies)  # sort by increasing energy
    keep = []  # always keep lowest-energy conformer
    discard = []

    for i in sort:
        this_tfd = tfd[i][np.asarray(keep, dtype=int)]
        # discard conformers within the tfd threshold
        if np.all(this_tfd >= tfd_thresh):
            keep.append(i)
        else:
            discard.append(i)

    # create a new molecule to hold the chosen conformers
    # this ensures proper conformer IDs and energy-based ordering
    new = Chem.Mol(mol)
    new.RemoveAllConformers()
    for i in keep:
        conf = mol.GetConformer(int(i))
        new.AddConformer(conf, assignId=True)

    return new

class CharMMMixin:
    def _seed(self, mol_name):
        charmm_configs = {
            "chignolin": {
                "toppar": [
                    "idp_rl/environments/environment_components/toppar/par_all36_prot.prm", 
                    "idp_rl/environments/environment_components/toppar/top_all36_prot.rtf",
                    "idp_rl/environments/environment_components/toppar/toppar_water_ions.str",
                ],
                "psf": "idp_rl/molecule_generation/chignolin/1uao.psf",
            }
        }

        charmm_config = charmm_configs[mol_name]
        openmm_toppar = app.CharmmParameterSet(*charmm_config["toppar"])
        openmm_psf = app.CharmmPsfFile(charmm_config["psf"])
        openmm_system = openmm_psf.createSystem(openmm_toppar)

        # TODO: test GPU version simulator
        integrator = openmm.VerletIntegrator(1.0)

        if torch.cuda.is_available():
            platform = openmm.Platform.getPlatformByName("CUDA")
            prop = dict(CudaPrecision="mixed")
            self.simulator = app.Simulation(openmm_psf.topology, openmm_system, integrator, platform, prop)
        else:
            platform = openmm.Platform.getPlatformByName("CPU")
            self.simulator = app.Simulation(openmm_psf.topology, openmm_system, integrator, platform)

    def _get_conformer_energy(self, mol: Chem.Mol, conf_id: int = None) -> float:
        """Returns the energy of the conformer with `conf_id` in `mol`.
        """
        if conf_id is None:
            conf_id = mol.GetNumConformers() - 1
        conf = mol.GetConformer(conf_id)

        positions = np_to_mm(conf.GetPositions())
        self.simulator.context.setPositions(positions)
        energy_kj = self.simulator.context.getState(getEnergy=True).getPotentialEnergy()
        energy_kcal = energy_kj.in_units_of(u.kilocalories_per_mole) # match RDKit/MMFF convention
        return energy_kcal._value

    def _get_conformer_energies(self, mol: Chem.Mol) -> List[float]:
        """Returns a list of energies for each conformer in `mol`.
        """
        return np.asarray([self._get_conformer_energy(mol, conf_id) for conf_id in range(mol.GetNumConformers())])

    def _optimize_conf(self, mol: Chem.Mol, conf_id: int = None, **kwargs):
        if conf_id is None:
            conf_id = mol.GetNumConformers() - 1
        conf = mol.GetConformer(conf_id)

        positions = np_to_mm(conf.GetPositions())
        self.simulator.context.setPositions(positions)
        self.simulator.minimizeEnergy(maxIterations=500)

        # CHARMM returns all of its positions in nm, so we have to convert back to Angstroms for RDKit
        optimized_positions_nm = self.simulator.context.getState(getPositions=True).getPositions()
        optimized_positions = optimized_positions_nm.in_units_of(u.angstrom) # match RDKit/MMFF convention

        for i, pos in enumerate(optimized_positions):
            conf.SetAtomPosition(i, Point3D(pos.x, pos.y, pos.z))

    def _calculate_normalizers(self, mol: Chem.Mol, num_confs: int = 200, pruning_thresh: float = 0.05):
        return calculate_normalizers(mol, self._optimize_conf, self._get_conformer_energies, num_confs, pruning_thresh)

    def _prune_conformers(self, mol: Chem.Mol, tfd_thresh: float) -> Chem.Mol:
        return prune_conformers(mol, self._get_conformer_energies, tfd_thresh)

class MMFFMixin:
    def _seed(self, mol_name):
        pass

    def _get_conformer_energy(self, mol: Chem.Mol, conf_id: int = None) -> float:
        """Returns the energy of the conformer with `conf_id` in `mol`.
        """
        if conf_id is None:
            conf_id = mol.GetNumConformers() - 1
        Chem.MMFFSanitizeMolecule(mol)
        mmff_props = Chem.MMFFGetMoleculeProperties(mol)
        ff = Chem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
        energy = ff.CalcEnergy()
        return energy

    def _get_conformer_energies(self, mol: Chem.Mol) -> List[float]:
        """Returns a list of energies for each conformer in `mol`.
        """
        return np.asarray([self._get_conformer_energy(mol, conf_id) for conf_id in range(mol.GetNumConformers())])

    def _optimize_conf(self, mol: Chem.Mol, conf_id: int = None, **kwargs):
        if conf_id is None:
            conf_id = mol.GetNumConformers() - 1
        Chem.MMFFOptimizeMolecule(mol, confId=conf_id, **kwargs)

    def _calculate_normalizers(self, mol: Chem.Mol, num_confs: int = 200, pruning_thresh: float = 0.05):
        return calculate_normalizers(mol, self._optimize_conf, self._get_conformer_energies, num_confs, pruning_thresh)

    def _prune_conformers(self, mol: Chem.Mol, tfd_thresh: float) -> Chem.Mol:
        return prune_conformers(mol, self._get_conformer_energies, tfd_thresh)
