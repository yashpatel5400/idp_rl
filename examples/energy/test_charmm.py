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

    SOLVENT_KWARGS = {
        'nonbondedMethod' : app.PME,
        'constraints' : None,
        'rigidWater' : True,
        'nonbondedCutoff' : 12.0 * u.angstroms,
    }

    testsystems = [
        ('1VII solvated', 'tests/step2_solvator.pdb', 'tests/step2_solvator.psf', ['toppar/par_all36_prot.prm', 'toppar/top_all36_prot.rtf','toppar/toppar_water_ions.str'], 'tests/step2.1_waterbox.prm', SOLVENT_KWARGS),
    ]

    for (name, pdb_filename, psf_filename, toppar_filenames, box_vectors_filename, system_kwargs) in testsystems:
        print('Testing %s' % name)
        compare_energies(name, pdb_filename, psf_filename, toppar_filenames, box_vectors_filename=box_vectors_filename, system_kwargs=system_kwargs)

def read_box_vectors(filename):
    """
    Read box vectors from CHARMM-GUI step2.1_waterbox.prm file that looks like:

     SET XTLTYPE  = CUBIC
     SET A = 80
     SET B = 80
     SET C = 80
     SET ALPHA = 90.0
     SET BETA  = 90.0
     SET GAMMA = 90.0
     SET FFTX     = 90
     SET FFTY     = 90
     SET FFTZ     = 90
     SET POSID = POT
     SET NEGID = CLA
     SET XCEN = 0
     SET YCEN = 0
     SET ZCEN = 0

    Returns
    -------
    box_vectors : simtk.unit.Quantity with shape [3,3] and units of Angstroms
        Box vectors
    """
    with open(filename, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            tokens = line.split()
            if tokens[1] == 'A':
                a = float(tokens[3]) * u.angstroms
            if tokens[1] == 'B':
                b = float(tokens[3]) * u.angstroms
            if tokens[1] == 'C':
                c = float(tokens[3]) * u.angstroms
            if tokens[1] == 'FFTX':
                fftx = int(tokens[3])
            if tokens[1] == 'FFTY':
                ffty = int(tokens[3])
            if tokens[1] == 'FFTZ':
                fftz = int(tokens[3])

    box_vectors = u.Quantity(np.zeros([3,3], np.float32), u.angstroms)
    SCALING = 1.1
    box_vectors[0][0] = a * SCALING
    box_vectors[1][1] = b * SCALING
    box_vectors[2][2] = c * SCALING
    return box_vectors

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
    """
    Compare energies between (pdb, psf, toppar) loaded via ParmEd and (pdb, ffxml) loaded by OpenMM ForceField

    Parameters
    ----------
    system_name : str
        Name of the test system
    pdb_filename : str
        Name of PDB file that should contain CRYST entry and PDB format compliant CONECT records for HETATM residues.
    psf_filename : str
        CHARMM PSF file
    toppar_filenames : list of CHARMM toppar filenames to load into CharmmParameterSet
        List of CHARMM toppar files
    box_vectors_filename : str, optional, default=None
        If specified, read box vectors from a file like step2.1_waterbox.prm
    system_kwargs : dict, optional, default=None
        Keyword arguments to pass to CharmmPsfFile.createSystem() and ForceField.CreateSystem() when constructing System objects for energy comparison
    tolerance : float, optional, default=1e-5
        Relative energy discrepancy tolerance
    units : simtk.unit.Unit
        Unit to use for energy comparison
    write_serialized_xml : bool, optional, default=False
        If True, will write out serialized System XML files for OpenMM systems to aid debugging.

    """

    is_periodic = True
    pdbfile = app.PDBFile(pdb_filename)

    # Read box vectors
    if box_vectors_filename:
        box_vectors = read_box_vectors(box_vectors_filename)
        pdbfile.topology.setPeriodicBoxVectors(box_vectors)
    else:
        box_vectors = pdbfile.topology.getPeriodicBoxVectors()

    # Load CHARMM system through OpenMM
    openmm_toppar = app.CharmmParameterSet(*toppar_filenames)
    openmm_psf = app.CharmmPsfFile(psf_filename)
    # Set box vectors
    if is_periodic:
        openmm_psf.setBox(box_vectors[0][0], box_vectors[1][1], box_vectors[2][2])
    openmm_system = openmm_psf.createSystem(openmm_toppar, **system_kwargs)
    openmm_total_energy = compute_potential(openmm_system, pdbfile.positions) / units

    print(f"Energy: {openmm_total_energy}")

if __name__ == '__main__':
    test_charmm()
