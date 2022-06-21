from idp_rl.utils import charmm_utils

import numpy as np
import parmed as pmd
import rdkit.Chem.AllChem as Chem

chignolin_psf_fn = "idp_rl/molecule_generation/chignolin/1uao.psf"
toppar_filenames = [
    "idp_rl/utils/toppar/par_all36_prot.prm", 
    "idp_rl/utils/toppar/top_all36_prot.rtf",
    "idp_rl/utils/toppar/toppar_water_ions.str"
]

def test_seed_charmm(mocker):
    charmm_utils.seed_charmm_simulator(chignolin_psf_fn, toppar_filenames)

def test_charmm_energy(mocker):
    chignolin_pdb_fn = "idp_rl/molecule_generation/chignolin/1uao.pdb"
    
    # test from RDKit and compare with ParmEd to ensure no issue with reading
    chignolin = Chem.rdmolfiles.MolFromPDBFile(chignolin_pdb_fn, removeHs=False)
    conf = chignolin.GetConformer(0)
    positions = conf.GetPositions()
    chignolin_energy = charmm_utils.charmm_energy(chignolin_psf_fn, toppar_filenames, positions)

    pmd_chignolin = pmd.load_file(chignolin_pdb_fn)
    pmd_np_pos = pmd_chignolin.positions._value
    pmd_chignolin_energy = charmm_utils.charmm_energy(chignolin_psf_fn, toppar_filenames, pmd_np_pos)

    assert(np.isclose(chignolin_energy._value, pmd_chignolin_energy._value))
