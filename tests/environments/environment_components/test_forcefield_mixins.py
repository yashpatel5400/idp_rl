from idp_rl.environments.environment_components.forcefield_mixins import CharMMMixin, MMFFMixin

import copy
import numpy as np
import openmm
import rdkit.Chem.AllChem as Chem

chignolin_psf_fn = "idp_rl/molecule_generation/chignolin/1uao.psf"
toppar_filenames = [
    "idp_rl/environments/environment_components/toppar/par_all36_prot.prm", 
    "idp_rl/environments/environment_components/toppar/top_all36_prot.rtf",
    "idp_rl/environments/environment_components/toppar/toppar_water_ions.str"
]

def load_chignolin():
    chignolin_pdb_fn = "idp_rl/molecule_generation/chignolin/1uao.pdb"
    chignolin = Chem.rdmolfiles.MolFromPDBFile(chignolin_pdb_fn, removeHs=False)
    return chignolin

def test_seed_charmm(mocker):
    charmm_sim = CharMMMixin()
    charmm_sim._seed(chignolin_psf_fn, toppar_filenames)
    
def test_charmm_energy(mocker):
    charmm_sim = CharMMMixin()
    charmm_sim._seed(chignolin_psf_fn, toppar_filenames)

    chignolin = load_chignolin()
    chignolin_energy = charmm_sim._get_conformer_energy(chignolin, 0)
    assert(np.isclose(chignolin_energy._value, 1881.096))

def test_charmm_opt(mocker):
    charmm_sim = CharMMMixin()
    charmm_sim._seed(chignolin_psf_fn, toppar_filenames)
    
    chignolin = load_chignolin()
    pre_chignolin_energy = charmm_sim._get_conformer_energy(chignolin, 0)
    charmm_sim._optimize_conf(chignolin, 0)
    opt_chignolin_energy = charmm_sim._get_conformer_energy(chignolin, 0)

    assert(opt_chignolin_energy._value < pre_chignolin_energy._value)    
    opt_thresh = 65 # optimization should minimize to a value < 65 (usually between 50-60)
    assert(opt_chignolin_energy._value < opt_thresh)

def test_charmm_mmff_compat(mocker):
    charmm_sim = CharMMMixin()
    charmm_sim._seed(chignolin_psf_fn, toppar_filenames)
    
    mmff_sim = MMFFMixin()
    mmff_sim._seed(chignolin_psf_fn, toppar_filenames)

    chignolin = load_chignolin()
    charmm_chignolin_energy = charmm_sim._get_conformer_energy(chignolin, 0)
    mmff_chignolin_energy = mmff_sim._get_conformer_energy(chignolin, 0)
    
    print(f"CHARMM: {charmm_chignolin_energy} vs MMFF: {mmff_chignolin_energy}")

    chignolin_charmm_cp = copy.deepcopy(chignolin)
    chignolin_mmff_cp = copy.deepcopy(chignolin)

    charmm_sim._optimize_conf(chignolin_charmm_cp, 0)
    opt_charmm_chignolin_energy = charmm_sim._get_conformer_energy(chignolin_charmm_cp, 0)

    mmff_sim._optimize_conf(chignolin_mmff_cp, 0)
    opt_mmff_chignolin_energy = mmff_sim._get_conformer_energy(chignolin_mmff_cp, 0)
    
    print(f"CHARMM: {opt_charmm_chignolin_energy} vs MMFF: {opt_mmff_chignolin_energy}")