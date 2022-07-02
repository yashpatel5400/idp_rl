from idp_rl.environments.environment_components.forcefield_mixins import CharMMMixin, MMFFMixin
from idp_rl.molecule_generation.generate_chignolin import generate_chignolin

import copy
import numpy as np
import openmm
import rdkit.Chem.AllChem as Chem

def test_seed_charmm(mocker):
    charmm_sim = CharMMMixin()
    charmm_sim._seed("chignolin")
    
def test_charmm_energy(mocker):
    charmm_sim = CharMMMixin()
    charmm_sim._seed("chignolin")

    chignolin = generate_chignolin()
    chignolin_energy = charmm_sim._get_conformer_energy(chignolin, 0)
    assert(np.isclose(chignolin_energy, 1881.096))

def test_charmm_opt(mocker):
    charmm_sim = CharMMMixin()
    charmm_sim._seed("chignolin")
    
    chignolin = generate_chignolin()
    pre_chignolin_energy = charmm_sim._get_conformer_energy(chignolin, 0)
    charmm_sim._optimize_conf(chignolin, 0)
    opt_chignolin_energy = charmm_sim._get_conformer_energy(chignolin, 0)

    assert(opt_chignolin_energy < pre_chignolin_energy)    
    opt_thresh = 65 # optimization should minimize to a value < 65 (usually between 50-60)
    assert(opt_chignolin_energy < opt_thresh)

def test_charmm_mmff_compat(mocker):
    charmm_sim = CharMMMixin()
    charmm_sim._seed("chignolin")
    
    mmff_sim = MMFFMixin()
    mmff_sim._seed("chignolin")

    chignolin = generate_chignolin()
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