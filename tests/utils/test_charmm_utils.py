from idp_rl.utils import charmm_utils
import numpy as np

def test_seed_charmm(mocker):
    chignolin_psf_fn = "idp_rl/molecule_generation/chignolin/1uao.psf"
    toppar_filenames = [
        "idp_rl/utils/toppar/par_all36_prot.prm", 
        "idp_rl/utils/toppar/top_all36_prot.rtf",
        "idp_rl/utils/toppar/toppar_water_ions.str"
    ]

    charmm_utils.seed_charmm_simulator(chignolin_psf_fn, toppar_filenames)