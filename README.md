HASTESM version 0.9

Written by Samuli Näppi and Tuomo Kalliokoski, Orion Pharma.
This software is meant to be run at Orion AWS cloud environment, but
you may find it useful nevertheless.

Additional software requirements:
You need also Schrödinger Suite (Phase/LigPrep and shape_matching) and
slurm. 

Hardware: 500 CPU cores for confgen / machine learning
prediction, 1 GPU for machine learning training and 40 CPU cores for
shape matching.

# Installing HASTESM from GitHub

Anaconda3 is recommended for the installation (create fresh new environment).

```
conda create -n hastesm-0.9 python=3.11 -y
conda activate hastesm-0.9
mamba install pytorch=2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install chemprop==2.0.4
mamba install pigz=2.6 -y
pip install git+https://github.com/username/repo-name.git
```

Edit file default_config.txt in the installation to match your environment.

## Running the software

Take a copy example_config.txt and edit it to your needs. Calculation can be started by typing:

```
hastesm -c my_copy_of_example_config.txt
```

## Updating the Package

To update to the latest version:

pip install --upgrade git+https://github.com/username/repo-name.git
```

## For Developers

If you're planning to develop or modify the package, you might want to install it in editable mode:

```
git clone https://github.com/username/repo-name.git
cd repo-name
pip install -e .
```

