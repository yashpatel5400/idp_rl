"""
Chignolin Generator
=================
"""
from rdkit import Chem

def generate_chignolin() -> Chem.Mol:
    """Generates chignolin molecule.
    """

    chignolin_pdb_fn = "idp_rl/molecule_generation/chignolin/1uao.pdb"
    chignolin = Chem.rdmolfiles.MolFromPDBFile(chignolin_pdb_fn, removeHs=False)
    Chem.SanitizeMol(chignolin)
    return chignolin

