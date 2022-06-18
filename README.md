![](https://raw.githubusercontent.com/deepmind/alphafold/main/imgs/header.jpg)

Intrinsically Disordered Proteins using RL (IDP RL)
========================================

IDP RL is a research study for investigating the use of RL to do conformer prediction on intrinsically disordered
proteins. 

[![Documentation Status](https://readthedocs.org/projects/conformer-rl/badge/?version=latest)](https://conformer-rl.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/conformer-rl.svg)](https://badge.fury.io/py/conformer-rl)

## Documentation
Documentation can be found at https://conformer-rl.readthedocs.io/.

## Platform Support
Since conformer-rl can be run within a Conda environment, it should work on all platforms (Windows, MacOS, Linux).

## Installation and Quick Start
Please see the documentation for [installation instructions](https://conformer-rl.readthedocs.io/en/latest/tutorial/install.html) and [getting started](https://conformer-rl.readthedocs.io/en/latest/tutorial/getting_started.html).

## Issues and Feature Requests
We are actively adding new features to this project and are open to all suggestions. If you believe you have encountered a bug, or if you have a feature that you would like to see implemented, please feel free to file an [issue](https://github.com/ZimmermanGroup/conformer-rl/issues).

## Developer Documentation
Pull requests are always welcome for suggestions to improve the code or to add additional features. We encourage new developers to document new features and write unit tests (if applicable). For more information on writing documentation and unit tests, see the [developer documentation](https://conformer-rl.readthedocs.io/en/latest/developer.html).

========================================
## Running on Great Lakes
Run the following code to do installation for now for Great Lakes:
```bash
conda create --name rl python=3.8
conda activate rl

python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
python -m pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
python -m pip install torch-geometric tensorboard rdkit gym stable-baselines3 py3Dmol
```