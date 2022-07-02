"""
Pre-built Environments
======================

This module contains several pre-built experiments. Each pre-built environment is created by overriding the following components:

* **Action Handler** refers to overriding of the :meth:`~idp_rl.environments.conformer_env.ConformerEnv._step` method of
  :class:`~idp_rl.environments.conformer_env.ConformerEnv`, 
  which determines how the molecule is modified given some action.
* **Reward Handler** refers to overriding of the :meth:`~idp_rl.environments.conformer_env.ConformerEnv._reward` method of
  :class:`~idp_rl.environments.conformer_env.ConformerEnv`, 
  which determines how the reward is calculated based on the current configuration of the molecule.
* **Observation Handler** refers to overriding of the :meth:`~idp_rl.environments.conformer_env.ConformerEnv._obs` method of
  :class:`~idp_rl.environments.conformer_env.ConformerEnv`, 
  which returns an observation object based on the current configuration of the molecule and is a compatible input for the neural net being used for training.

All pre-built environments inherit from :class:`~idp_rl.environments.conformer_env.ConformerEnv` and share the same constructor.

"""

from idp_rl.environments.conformer_env import ConformerEnv
from idp_rl.environments.curriculum_conformer_env import CurriculumConformerEnv
from idp_rl.environments.environment_components.action_mixins import ContinuousActionMixin, DiscreteActionMixin
from idp_rl.environments.environment_components.reward_mixins import GibbsRewardMixin, GibbsPruningRewardMixin, GibbsEndPruningRewardMixin, GibbsLogPruningRewardMixin
from idp_rl.environments.environment_components.obs_mixins import GraphObsMixin, AtomCoordsTypeGraphObsMixin
from idp_rl.environments.environment_components.forcefield_mixins import CharMMMixin, MMFFMixin




class DiscreteActionEnv(DiscreteActionMixin, GraphObsMixin, MMFFMixin, ConformerEnv):
    """
    * Action Handler: :class:`~idp_rl.environments.environment_components.action_mixins.DiscreteActionMixin`
    * Reward Handler: default reward handler from :class:`~idp_rl.environments.conformer_env.ConformerEnv`
    * Observation Handler: :class:`~idp_rl.environments.environment_components.obs_mixins.GraphObsMixin`
    """
    pass


class GibbsScoreEnv(GibbsRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, MMFFMixin, ConformerEnv):
    """
    * Action Handler: :class:`~idp_rl.environments.environment_components.action_mixins.DiscreteActionMixin`
    * Reward Handler: :class:`~idp_rl.environments.environment_components.reward_mixins.GibbsRewardMixin`
    * Observation Handler: :class:`~idp_rl.environments.environment_components.obs_mixins.AtomCoordsTypeGraphObsMixin`
    """
    pass


class GibbsScorePruningEnv(GibbsPruningRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, MMFFMixin, ConformerEnv):
    """
    * Action Handler: :class:`~idp_rl.environments.environment_components.action_mixins.DiscreteActionMixin`
    * Reward Handler: :class:`~idp_rl.environments.environment_components.reward_mixins.GibbsPruningRewardMixin`
    * Observation Handler: :class:`~idp_rl.environments.environment_components.obs_mixins.AtomCoordsTypeGraphObsMixin`
    """
    pass


class GibbsScoreEndPruningEnv(GibbsEndPruningRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, MMFFMixin, ConformerEnv):
    """
    * Action Handler: :class:`~idp_rl.environments.environment_components.action_mixins.DiscreteActionMixin`
    * Reward Handler: :class:`~idp_rl.environments.environment_components.reward_mixins.GibbsEndPruningRewardMixin`
    * Observation Handler: :class:`~idp_rl.environments.environment_components.obs_mixins.AtomCoordsTypeGraphObsMixin`
    """
    pass


class GibbsScorePruningEnvCharmm(GibbsPruningRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, CharMMMixin, ConformerEnv):
    """
    * Action Handler: :class:`~idp_rl.environments.environment_components.action_mixins.DiscreteActionMixin`
    * Reward Handler: :class:`~idp_rl.environments.environment_components.reward_mixins.GibbsPruningRewardMixin`
    * Observation Handler: :class:`~idp_rl.environments.environment_components.obs_mixins.AtomCoordsTypeGraphObsMixin`
    """
    pass


class GibbsScoreLogPruningEnv(GibbsLogPruningRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, MMFFMixin, ConformerEnv):
    """
    * Action Handler: :class:`~idp_rl.environments.environment_components.action_mixins.DiscreteActionMixin`
    * Reward Handler: :class:`~idp_rl.environments.environment_components.reward_mixins.GibbsLogPruningRewardMixin`
    * Observation Handler: :class:`~idp_rl.environments.environment_components.obs_mixins.AtomCoordsTypeGraphObsMixin`
    """
    pass

class GibbsScorePruningCurriculumEnv(GibbsPruningRewardMixin, AtomCoordsTypeGraphObsMixin, DiscreteActionMixin, MMFFMixin, CurriculumConformerEnv):
    """Same handlers as the :class:`~idp_rl.environments.environment.GibbsScorePruningEnv` but with support for curriculum learning."""

class GibbsScoreLogPruningCurriculumEnv(GibbsLogPruningRewardMixin, DiscreteActionMixin, AtomCoordsTypeGraphObsMixin, MMFFMixin, CurriculumConformerEnv):
    """Same handlers as the :class:`~idp_rl.environments.environment.GibbsScoreLogPruningEnv` but with support for curriculum learning."""