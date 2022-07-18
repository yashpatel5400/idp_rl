import copy
import os
import random
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from idp_rl import utils
from idp_rl.agents import PPOAgent
from idp_rl.config import Config
from idp_rl.environments import Task
from idp_rl.models import RTGN
from idp_rl.environments.environment_components.forcefield_mixins import CharMMMixin

from idp_rl.molecule_generation.generate_chignolin import generate_chignolin
from idp_rl.molecule_generation.generate_molecule_config import config_from_rdkit

import logging
logging.basicConfig(level=logging.DEBUG)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    os.environ["MP_RANK"] = str(rank)

    utils.set_one_thread()

    # configure molecule
    mol_name = "chignolin"
    filename = f"{mol_name}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            mol_config = pickle.load(file)
    else:
        mol = generate_chignolin()
        ff_mixin = CharMMMixin()
        ff_mixin._seed(mol_name, rank)
        mol_config = config_from_rdkit(mol, num_conformers=200, calc_normalizers=True, save_file=mol_name, ff_mixin=ff_mixin)

    # create agent config and set environment
    seed = np.random.randint(int(1e5))

    config = Config()
    config.tag = 'chignolin'

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
    config.rollout_length = 50
    config.recurrence = 5
    config.optimization_epochs = 4
    config.max_steps = 200000
    config.mini_batch_size = 60 // world_size

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

    # Neural Network
    setup(rank, world_size)
    config.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    config.network = DDP(RTGN(6, 128, edge_dim=6, node_dim=5).to(config.device))

    total_envs = 24
    envs_per_node = (total_envs // world_size)
    config.train_env = Task('GibbsScorePruningEnvCharmm-v0', concurrency=True, num_envs=envs_per_node, seed=seed, pt_rank=rank, mol_config=mol_config)

    torch.manual_seed(envs_per_node * rank + seed)

    agent = PPOAgent(config)
    agent.run_steps()

if __name__ == '__main__':
    world_size = 6

    mp.spawn(main,
        args=(world_size,),
        nprocs=world_size,
        join=True)