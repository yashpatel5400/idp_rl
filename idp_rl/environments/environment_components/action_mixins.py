"""
Action_mixins
=============

Pre-built action handlers.
"""

from rdkit import Chem
from typing import List

class ContinuousActionMixin:
    """For each torsion of the molecule, modifies the torsion given an angle from a continuous range.
    """

    def _step(self, action: List[float]) -> None:
        """Sets the torsion angles of the molecule.

        Parameters
        ----------
        action : list of float
            Each element of `action` specifies the angle (in degrees) to set the angle of the corresponding
            torsion in the molecule.

        Notes
        -----
        Logged parameters:

        * conf: the current generated conformer is saved to the episodic mol object.
        """
        conf = self.conf
        for idx, tors in enumerate(self.nonring_original):
            Chem.rdMolTransforms.SetDihedralDeg(conf, *tors, float(action[idx]))
        self._optimize_conf(self.mol, conf_id=self.mol.GetNumConformers() - 1)
        self.episode_info['mol'].AddConformer(self.conf, assignId=True, maxIters=500, nonBondedThresh=10.)
    
class DiscreteActionMixin:
    """For each torsion of the molecule, modifies the torsion given an angle from a discrete set of possible angles.
    """

    def _step(self, action: List[int]) -> None:
        """Sets the torsion angles of the molecule.

        Parameters
        ----------
        action : list of int between 0 and 5
            For each element of `action`, sets the corresponding torsion angle to 60 times the element degrees.
        
        Notes
        -----
        Logged parameters:
        
        * conf: the current generated conformer is saved to the episodic mol object.
        """
        for idx, tors in enumerate(self.nonring_original):
            ang = -180 + 60 * action[idx]
            Chem.rdMolTransforms.SetDihedralDeg(self.conf, *tors, float(ang))
        self._optimize_conf(self.mol, conf_id=self.mol.GetNumConformers() - 1, maxIters=500, nonBondedThresh=10.)
        self.episode_info['mol'].AddConformer(self.conf, assignId=True)