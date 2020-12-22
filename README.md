# EnKFDiffeo

This repository contains the Python code used in [this paper](http://arxiv.org/).

# Dependencies & Installation

To install, run the following commands in `enkf_landmarks`.
```
enkf_landmarks> poetry shell
enkf_landmarks> poetry install
```
Issue the following command to verify everything has gone well:
```
enkf_landmarks> python src/verify_pytorch.py
```
which should produce:
```
Remember to run 'export GLOO_SOCKET_IFNAME=<your primary interface>!
Rank: 0 of 4
Rank: 1 of 4
Rank: 2 of 4
Rank: 3 of 4

That took 2.3776447772979736 seconds
```

You should also run 
```
enkf_landmarks> python src/example_lddmm.py
```
which generates `example_lddmm.pdf` to make sure PyKeops is working as expected.

# Overview of the repository

```
enkf_landmarks
│   README.md
└───src
│   └─── enkf.py                            # main class defining the EnKF
│   └─── example_lddmm.py                   # script to demonstrate the PyTorch forward operator
│   └─── lddmm.py                           # defines the forward operator written in PyTorch
│   └─── manufacture_shape_data.py          # script generates new targets from random initial momentum
│   └─── run_enkf.py                        # run the enkf on the targets in the data directory
│   └─── run_regularisation_experiments.py  # run a parameter study on the regularisation parameter
│   └─── utils.py
│   └─── verify_gloo_pytorch.py             # run this to test distributed Pytorch
└───data   # contains the targets we use in the `run_*.py` scripts
    └───LANDMARKS=10
    └───LANDMARKS=50
    └───LANDMARKS=150
```

# Reproducing the figures in the paper

Remember to activate the virtual environment before running the commands below.

  - Figure 1, 3-8: `python run_enkf.py` creates a `RESULTS_ENKF` folder and dumps all the results therein.
  - Figure 2: `python run_regularisation_experiments.py` creates the directory `RESULTS_REGULARISATION_EXPERIMENTS` with the figures.

# License

See `LICENSE.txt`.

# Todo

 - Make it work for curves or images.
 - Clean up `numpy` usage.
 - Use `pathlib` everywhere, what a mess this is. Honestly mate.
