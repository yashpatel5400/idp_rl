import sys
import random
from simtk import openmm, unit
from simtk.openmm import app
import numpy as np
import mdtraj
import matplotlib.pyplot as plt

# load files
psf = app.CharmmPsfFile("init.psf")
pdbfile = app.PDBFile("init.pdb")
params = app.CharmmParameterSet("toppar/top_all36_prot.rtf", "toppar/par_all36m_prot.prm", "toppar/toppar_water_ions.str")
positions = pdbfile.getPositions(asNumpy=True)

# system
psf.setBox(39.0*unit.angstroms, 39.0*unit.angstroms, 39.0*unit.angstroms)

system = psf.createSystem(params,
    nonbondedMethod=app.PME, 
    nonbondedCutoff=1.2*unit.nanometers,
    implicitSolvent=None,
    constraints=app.HBonds,
    rigidWater=True, 
    verbose=True, 
    ewaldErrorTolerance=0.0005)

barostat = openmm.MonteCarloBarostat(1.0*unit.bar, 300.0*unit.kelvin)
system.addForce(barostat)

# integrator
integrator = openmm.LangevinIntegrator(300.0*unit.kelvin,   # Temperature of head bath
                                       1.0/unit.picosecond, # Friction coefficient
                                       0.002*unit.picoseconds) # Time step
random.seed()
integrator.setRandomNumberSeed(random.randint(1,100000))

# platform
platform = openmm.Platform.getPlatformByName("CPU")
# platform = openmm.Platform.getPlatformByName("CUDA")
# prop = dict(CudaPrecision="mixed")

# Build simulation context
simulation = app.Simulation(psf.topology, system, integrator, platform)
# simulation = app.Simulation(psf.topology, system, integrator, platform, prop)
simulation.context.setPositions(positions)

# initial system energy
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
simulation.minimizeEnergy()
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

# simulation
reportInterval = 10
nstep = 1000
simulation.reporters.append(app.StateDataReporter(sys.stdout, reportInterval, step=True, time=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=nstep, separator="\t"))
simulation.reporters.append(mdtraj.reporters.HDF5Reporter("run.h5", reportInterval, coordinates=True, time=True, cell=True, potentialEnergy=True, temperature=True))
simulation.reporters.append(mdtraj.reporters.DCDReporter("run.dcd", reportInterval))
simulation.step(nstep)

# save checkpoint
with open("run.chk", "wb") as f:
    f.write(simulation.context.createCheckpoint())

del simulation # Make sure to close all files

native = mdtraj.load("1uao.pdb")
traj = mdtraj.load("run.dcd", top="init.psf")
traj = traj.atom_slice(traj.topology.select("protein"))
traj.superpose(native)

rmsd = mdtraj.rmsd(traj, native) * 10.0
plt.plot(rmsd)
plt.xlabel("step")
plt.ylabel("RMSD (Angstrom)")
plt.savefig("analysis.png", dpi=350)