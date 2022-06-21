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

    def conf_energy(self, positions: openmm.unit.quantity.Quantity):
        self.simulator.context.setPositions(positions)
        energy = self.simulator.context.getState(getEnergy=True).getPotentialEnergy()
        return energy

    def optimize_conf(self, positions: openmm.unit.quantity.Quantity):
        self.simulator.context.setPositions(positions)
        self.simulator.minimizeEnergy(maxIterations=500)
        optimized_positions = self.simulator.context.getState(getPositions=True).getPositions()
        return optimized_positions