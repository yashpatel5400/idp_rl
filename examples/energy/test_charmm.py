import openmm
import openmm.app as app
import simtk.unit as u
import numpy as np

def test_charmm():
    """
    Test CHARMM ffxml conversion by computing energy discrepancies between (pdb, psf, toppar) loaded via ParmEd and (pdb, ffxml) loaded via OpenMM ForceField

    Parameters
    ----------
    write_serialized_xml : bool, optional, default=False
        If True, will write out serialized System XML files for OpenMM systems to aid debugging.

    """

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
        ('1VII protein', 'tests/step1_pdbreader.pdb', 'tests/step1_pdbreader.psf', ['toppar/par_all36_prot.prm', 'toppar/top_all36_prot.rtf','toppar/toppar_water_ions.str'], None, VACUUM_KWARGS),
    ]

    for (name, pdb_filename, psf_filename, toppar_filenames, box_vectors_filename, system_kwargs) in testsystems:
        compare_energies(name, pdb_filename, psf_filename, toppar_filenames, box_vectors_filename=box_vectors_filename, system_kwargs=system_kwargs)

def compute_potential(system, positions):
    """
    Compute potential energy

    Parameters
    ----------
    system : simtk.openmm.System
        System
    positions : simtk.unit.Quantity of shape [nparticles,3] with units compatible with angstroms
        Positions

    Returns
    -------
    potential : simtk.unit.Quantity with units of kJ/mol
        The potential energy

    """
    integrator = openmm.VerletIntegrator(1.0)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    potential = context.getState(getEnergy=True).getPotentialEnergy()
    del context, integrator
    return potential

def compare_energies(system_name, pdb_filename, psf_filename, toppar_filenames, box_vectors_filename=None, system_kwargs=None, tolerance=1e-5, units=u.kilojoules_per_mole, write_serialized_xml=False):
    pdbfile = app.PDBFile(pdb_filename)
    openmm_toppar = app.CharmmParameterSet(*toppar_filenames)
    openmm_psf = app.CharmmPsfFile(psf_filename)
    openmm_system = openmm_psf.createSystem(openmm_toppar, **system_kwargs)
    openmm_total_energy = compute_potential(openmm_system, pdbfile.positions) / units
    print(f"Energy: {openmm_total_energy}")

if __name__ == '__main__':
    test_charmm()
