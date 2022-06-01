import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing
import logging
import torch
import pandas as pd
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance
import torch_geometric.nn as gnn

import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from idp_rl.utils import *
from idp_rl.models import *
from idp_rl.environments import Task

random.seed(4)
np.random.seed(4)
torch.manual_seed(4)

from concurrent.futures import ProcessPoolExecutor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loaded_policy(model, env):
    num_envs = 1
    single_process = (num_envs == 1)

    env = Task(env, seed=random.randint(0,7e4), num_envs=num_envs, single_process=single_process)
    state = env.reset()
    total_reward = 0
    start = True
    done = False
    step = 0
    while not done:
        with torch.no_grad():
            if start:
                prediction, rstates = model(state)
                start = False
            else:
                prediction, rstates = model(state, rstates)

        choice = prediction['a']
        step += 1
        print('step', step)
        state, rew, done_, info = env.step(to_np(choice))
        total_reward += float(rew)
        print('rew', rew)
        print('total_reward', total_reward)

        done = bool(done_)

    if isinstance(info, tuple):
        for i, info_ in enumerate(info):
            print('episodic_return', info_['episodic_return'])
    else:
        print('episodic_return', info['episodic_return'])
    return total_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run batch training')
    parser.add_argument('--cfg')
    args = parser.parse_args()
    
    with open(args.cfg, "r") as f:
        config = json.load(f)

    model = RTGNBatch(6, 128, edge_dim=6, point_dim=5)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))

    outputs = []
    times = []
    for i in range(10):
        start = time.time()
        output = loaded_policy(model, config["env_name"])
        print('output', output)
        end = time.time()
        outputs.append(output)
        times.append(end - start)
    print('outputs', outputs)
    print('times', times)
