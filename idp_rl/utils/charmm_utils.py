"""
CHARMM Utilities
===================

CHARMM utility functions (energy calculations and molecular relaxations).
"""
import torch
import openmm
import openmm.app as app
import openmm.unit as u
import numpy as np

from typing import Tuple, List

charmm_simulator = None

def seed_charmm_simulator(psf_fn: str, toppar_filenames: List[str]):
    global charmm_simulator
    if charmm_simulator is not None:
        return

    openmm_toppar = app.CharmmParameterSet(*toppar_filenames)
    openmm_psf = app.CharmmPsfFile(psf_fn)
    openmm_system = openmm_psf.createSystem(openmm_toppar)

    # TODO: test GPU version simulator
    integrator = openmm.VerletIntegrator(1.0)

    if torch.cuda.is_available():
        platform = openmm.Platform.getPlatformByName("CUDA")
        prop = dict(CudaPrecision="mixed")
        charmm_simulator = app.Simulation(psf.topology, system, integrator, platform, prop)
    else:
        platform = openmm.Platform.getPlatformByName("CPU")
        charmm_simulator = app.Simulation(openmm_psf.topology, openmm_system, integrator, platform)
    
def np_to_mm(arr: np.ndarray, unit: openmm.unit=openmm.unit.angstrom):
    wrapped_val = openmm.unit.quantity.Quantity(arr, unit)
    return wrapped_val

def charmm_energy(psf_fn: str, toppar_filenames: List[str], positions: np.ndarray):
    seed_charmm_simulator(psf_fn, toppar_filenames)
    global charmm_simulator
    mm_pos = np_to_mm(positions)
    charmm_simulator.context.setPositions(mm_pos)
    energy = charmm_simulator.context.getState(getEnergy=True).getPotentialEnergy()
    return energy

def charmm_optimize_conf(psf_fn: str, toppar_filenames: List[str], positions: np.ndarray):
    seed_charmm_simulator(psf_fn, toppar_filenames)
    global charmm_simulator
    mm_pos = np_to_mm(positions)
    charmm_simulator.context.setPositions(mm_pos)
    charmm_simulator.minimizeEnergy(maxIterations=500)
    optimized_positions = charmm_simulator.context.getState(getPositions=True).getPositions()
    return optimized_positions