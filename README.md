Intrinsically Disordered Proteins using RL (IDP RL)
========================================
![](https://raw.githubusercontent.com/deepmind/alphafold/main/imgs/header.jpg)

[![](https://img.shields.io/badge/docs-master-blue.svg)](https://prob-ml.github.io/bliss/)
![tests](https://github.com/prob-ml/bliss/workflows/tests/badge.svg)
[![codecov.io](https://codecov.io/gh/prob-ml/bliss/branch/master/graphs/badge.svg?branch=master&token=Jgzv0gn3rA)](http://codecov.io/github/prob-ml/bliss?branch=master)
![case studies](https://github.com/prob-ml/bliss/actions/workflows/case_studies.yml/badge.svg)

# Introduction

IDP RL is a research study for investigating the use of RL to do conformer prediction on intrinsically disordered
proteins. 

# Installation

1. To use and install `idp_rl` you first need to install Python. Anaconda is the easiest way:
```
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
```

2. For Python dependencies, you then need to install [poetry](https://python-poetry.org/docs/):
```
curl -sSL https://install.python-poetry.org | python3 -
```

3. Now download the idp_rl repo:
```
git clone https://github.com/yashpatel5400/idp_rl.git
```

5. To create a poetry environment with the `idp_rl` dependencies satisified, run
```
cd idp_rl
poetry install
poetry shell
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
```

6. Verify that bliss is installed correctly by running the tests both on your CPU (default) and on your GPU:
```
pytest
pytest --gpu
```

# Latest updates
- The original TorsionNet repo is going through a massive overhaul for IDP RL


# References

Tarun Gogineni, Ziping Xu, Exequiel Punzalan, Runxuan Jiang, Joshua Kammeraad, Ambuj Tewari, and Paul Zimmerman. *TorsionNet: A Reinforcement Learning Approach to Sequential Conformer Search* [https://arxiv.org/abs/2006.07078](https://arxiv.org/abs/2006.07078), 2020.

---

## File Layout

* The `TorsionNet` directory contains scripts for running each of the three experiments mentioned in the TorsionNet paper: {alkane, lignin, and t_chains}. For more details on running the scripts, see the [run](##Run) section below.
* Scripts for generating the molecule files for each of the three experiments are located in the `TorsionNet` directory and are named `generate_{branched_alkanes, lignin, t_chain}.py` corresponding to each experiment.
* The agents, models, and environments are stored in the directory `TorsionNet/main`.
    * `TorsionNet/main/agents` contains implementations for the custom agents. The main agent used is PPO, which is stored in the file `PPO_recurrent_agent.py`. Some of the code for the agents is roughly based off of the RL framework [DeepRL](https://github.com/ShangtongZhang/DeepRL).
    * `TorsionNet/main/environments` contains implementations for the reinforcement learning environments used in each experiment. Most environments are stored in `graphenvironments.py`.
    * The file `models.py` in `TorsionNet/main` contains the implementation for the neural network used in most experiments, RTGNBatch.
* Pre-trained model parameters for each of the three experiments are stored in `TorsionNet/trained_models`.

## Run

Train and evaluation python scripts are located in the `TorsionNet` directory for all experiments: {alkane, lignin, t_chain}.

Scripts for training agents for each experiment are named `train_[experiment_name].py`. For example, to run the lignin experiments, run
 ```
 cd TorsionNet/
 python train_lignin.py
 ```
NOTE: for training the alkane environment, unzip the file `huge_hc_set.zip` first.

Model parameters are saved in the `TorsionNet/data` directory. Tensorboard is available to monitor the training process:
```
cd TorsionNet/
tensorboard --logdir tf_log/
```

Evaluation scripts are available for each of the experiments and are named `eval_[experiment name].py`. To run the evaluation script, we provide sample pre-trained model parameters. If training from scratch, first replace the path parameter in the "torch.load" function in the script with the path of the model weights that perform best on the validation environment. This can be checked via tensorboard. Model weights are stored at the same cadence as validation runs. After replacing the path parameter, the eval script can be run e.g.
```
cd TorsionNet/
python eval_lignin.py
```

## Results

This is a best effort reproduction of our implementation. There may be some nondeterminism.
