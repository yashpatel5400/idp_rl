import copy
import os
import pickle
import numpy as np
import torch

from idp_rl import utils
from idp_rl.agents import PPOAgent
from idp_rl.config import Config
from idp_rl.environments import Task
from idp_rl.models import RTGN
from idp_rl.environments.environment_components.forcefield_mixins import CharMMMixin

from idp_rl.molecule_generation.generate_chignolin import generate_chignolin
from idp_rl.molecule_generation.generate_molecule_config import config_from_rdkit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    utils.set_one_thread()

    # configure molecule
    mol = generate_chignolin()
    mol_name = "chignolin"
    filename = f"{mol_name}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            mol_config = pickle.load(file)
    else:
        ff_mixin = CharMMMixin()
        ff_mixin._seed(mol_name)
        mol_config = config_from_rdkit(mol, num_conformers=200, calc_normalizers=True, save_file=mol_name, ff_mixin=ff_mixin)

    # create agent config and set environment
    config = Config()
    config.tag = 'chignolin'
    config.train_env = Task('GibbsScorePruningEnvCharmm-v0', concurrency=True, num_envs=5, mol_config=mol_config)

    # Neural Network
    config.network = RTGN(6, 128, edge_dim=6, node_dim=5).to(device)

    # Logging Parameters
    config.save_interval = 20000
    config.data_dir = 'data'
    config.use_tensorboard = True

    # Set up evaluation
    eval_mol_config = copy.deepcopy(mol_config) # config_from_rdkit(mol, calc_normalizers=True, save_file=f'{mol_name}_eval')
    config.eval_env = Task('GibbsScorePruningEnvCharmm-v0', num_envs=1, mol_config=eval_mol_config)
    config.eval_interval = 20000
    config.eval_episodes = 2

    # Batch Hyperparameters
    config.rollout_length = 20
    config.recurrence = 5
    config.optimization_epochs = 4
    config.max_steps = 200000
    config.mini_batch_size = 50

    # Training Hyperparameters
    lr = 5e-6 * np.sqrt(10)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.discount = 0.9999
    config.use_gae = True
    config.gae_lambda = 0.95
    config.entropy_weight = 0.001
    config.value_loss_weight = 0.25
    config.gradient_clip = 0.5
    config.ppo_ratio_clip = 0.2

    agent = PPOAgent(config)
    agent.run_steps()