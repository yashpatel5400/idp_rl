from idp_rl.utils import charmm_utils

import numpy as np
import openmm
import rdkit.Chem.AllChem as Chem
from rdkit.Geometry import Point3D

chignolin_psf_fn = "idp_rl/molecule_generation/chignolin/1uao.psf"
toppar_filenames = [
    "idp_rl/utils/toppar/par_all36_prot.prm", 
    "idp_rl/utils/toppar/top_all36_prot.rtf",
    "idp_rl/utils/toppar/toppar_water_ions.str"
]

def test_seed_charmm(mocker):
    charmm_utils.CharmmSim(chignolin_psf_fn, toppar_filenames)

def test_charmm_energy(mocker):
    charmm_sim = charmm_utils.CharmmSim(chignolin_psf_fn, toppar_filenames)
    chignolin_pdb_fn = "idp_rl/molecule_generation/chignolin/1uao.pdb"
    
    # test from RDKit and compare with ParmEd to ensure no issue with reading
    chignolin = Chem.rdmolfiles.MolFromPDBFile(chignolin_pdb_fn, removeHs=False)
    conf = chignolin.GetConformer(0)
    chignolin_energy = charmm_sim.conf_energy(conf)

    assert(np.isclose(chignolin_energy._value, 1881.096))

def test_charmm_opt(mocker):
    charmm_sim = charmm_utils.CharmmSim(chignolin_psf_fn, toppar_filenames)
    
    # test from RDKit and compare with ParmEd to ensure no issue with reading
    chignolin_pdb_fn = "idp_rl/molecule_generation/chignolin/1uao.pdb"
    chignolin = Chem.rdmolfiles.MolFromPDBFile(chignolin_pdb_fn, removeHs=False)
    conf = chignolin.GetConformer(0)
    
    pre_chignolin_energy = charmm_sim.conf_energy(conf)

    optimized_pos = charmm_sim.optimize_conf(conf)
    for i, pos in enumerate(optimized_pos):
        conf.SetAtomPosition(i, Point3D(pos.x, pos.y, pos.z))
    opt_chignolin_energy = charmm_sim.conf_energy(conf)

    assert(opt_chignolin_energy._value < pre_chignolin_energy._value)    
    opt_thresh = 65 # optimization should minimize to a value < 65 (usually between 50-60)
    assert(opt_chignolin_energy._value < opt_thresh)