import numpy as np
import torch
import random

from idp_rl import utils
from idp_rl.agents import PPOAgent
from idp_rl.config import Config
from idp_rl.environments import Task
from idp_rl.models import RTGN

from idp_rl.molecule_generation.generate_lignin import generate_lignin
from idp_rl.molecule_generation.generate_molecule_config import config_from_rdkit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
import pickle
logging.basicConfig(level=logging.DEBUG)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    utils.set_one_thread()

    # configure molecule
    mol = generate_lignin(3)
    with open("lignin.pkl", "rb") as f:
        mol_config = pickle.load(f)
    # mol_config = config_from_rdkit(mol, num_conformers=100, calc_normalizers=True, save_file='lignin')

    # create agent config and set environment
    config = Config()
    config.tag = 'example2'
    config.train_env = Task('GibbsScorePruningEnv-v0', concurrency=True, num_envs=1, mol_config=mol_config)

    # Neural Network
    config.network = RTGN(6, 128, edge_dim=6, node_dim=5).to(device)

    # Logging Parameters
    config.save_interval = 20000
    config.data_dir = 'data'
    config.use_tensorboard = True

    # Set up evaluation
    eval_mol = generate_lignin(4)
    with open("lignin_eval.pkl", "rb") as f:
        eval_mol_config = pickle.load(f)
    # eval_mol_config = config_from_rdkit(mol, num_conformers=100, calc_normalizers=True, save_file='lignin_eval')
    config.eval_env = Task('GibbsScorePruningEnv-v0', num_envs=1, mol_config=eval_mol_config)
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