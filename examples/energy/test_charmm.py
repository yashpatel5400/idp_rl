import openmm
import openmm.app as app
import simtk.unit as u
import numpy as np

def test_charmm():
    # SOLVENT_KWARGS = {
    #     'nonbondedMethod' : app.PME,
    #     'constraints' : None,
    #     'rigidWater' : True,
    #     'nonbondedCutoff' : 12.0 * u.angstroms,
    # }

    VACUUM_KWARGS = {
        'nonbondedMethod' : app.NoCutoff,
        'constraints' : None,
    }

    testsystems = [
        ('1VII protein', 'tests/step1_pdbreader.pdb', 'tests/step1_pdbreader.psf', ['toppar/par_all36_prot.prm', 'toppar/top_all36_prot.rtf','toppar/toppar_water_ions.str'], VACUUM_KWARGS),
    ]

    for (name, pdb_filename, psf_filename, toppar_filenames, system_kwargs) in testsystems:
        compare_energies(name, pdb_filename, psf_filename, toppar_filenames, system_kwargs=system_kwargs)

def compare_energies(system_name, pdb_filename, psf_filename, toppar_filenames, system_kwargs=None, tolerance=1e-5, units=u.kilojoules_per_mole, write_serialized_xml=False):
    pdbfile = app.PDBFile(pdb_filename)
    openmm_toppar = app.CharmmParameterSet(*toppar_filenames)
    openmm_psf = app.CharmmPsfFile(psf_filename)
    openmm_system = openmm_psf.createSystem(openmm_toppar, **system_kwargs)

    integrator = openmm.VerletIntegrator(1.0)
    platform = openmm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(openmm_psf.topology, openmm_system, integrator, platform)
    simulation.context.setPositions(pdbfile.positions)

    print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
    simulation.minimizeEnergy()
    print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

if __name__ == '__main__':
    test_charmm()
