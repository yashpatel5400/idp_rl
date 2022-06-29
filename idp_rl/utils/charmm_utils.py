"""
CHARMM Utilities
===================

CHARMM utility functions (energy calculations and molecular relaxations).
"""
import torch
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u
import rdkit.Chem.AllChem as Chem

from typing import Tuple, List

def np_to_mm(arr: np.ndarray, unit: openmm.unit=u.angstrom):
    wrapped_val = openmm.unit.quantity.Quantity(arr, unit)
    return wrapped_val

class CharmmSim:
    def __init__(self, psf_fn: str, toppar_filenames: List[str]):
        openmm_toppar = app.CharmmParameterSet(*toppar_filenames)
        openmm_psf = app.CharmmPsfFile(psf_fn)
        openmm_system = openmm_psf.createSystem(openmm_toppar)

        # TODO: test GPU version simulator
        integrator = openmm.VerletIntegrator(1.0)

        if torch.cuda.is_available():
            platform = openmm.Platform.getPlatformByName("CUDA")
            prop = dict(CudaPrecision="mixed")
            self.simulator = app.Simulation(psf.topology, system, integrator, platform, prop)
        else:
            platform = openmm.Platform.getPlatformByName("CPU")
            self.simulator = app.Simulation(openmm_psf.topology, openmm_system, integrator, platform)

    def conf_energy(self, conf: Chem.rdchem.Conformer):
        positions = np_to_mm(conf.GetPositions())
        self.simulator.context.setPositions(positions)
        energy = self.simulator.context.getState(getEnergy=True).getPotentialEnergy()
        return energy

    def optimize_conf(self, conf: Chem.rdchem.Conformer):
        positions = np_to_mm(conf.GetPositions())
        self.simulator.context.setPositions(positions)
        self.simulator.minimizeEnergy(maxIterations=500)

        # CHARMM returns all of its positions in nm, so we have to convert back to Angstroms for RDKit
        optimized_positions_nm = self.simulator.context.getState(getPositions=True).getPositions()
        optimized_positions = optimized_positions_nm.in_units_of(u.angstrom)
        return optimized_positions