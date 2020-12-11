# EnKFDiffeo

This repository contains the Python code used in [this paper](http://arxiv.org/).

# Dependencies & Installation

To install we need  [this paper](http://arxiv.org/). After installation of this,
run the following in `enkf_landmarks`.
```
enkf_landmarks>  poetry shell
enkf_landmarks>  poetry install
```
Issue the following command to verify everything's gone well:
```
enkf_landmarks>  python src/verify_pytorch.py
```
which should produce:
```
Remember to run 'export GLOO_SOCKET_IFNAME=<your primary interface>!
Rank: 0 of 4
Rank: 1 of 4
Rank: 2 of 4
Rank: 3 of 4

That took 2.3776447772979736 seconds```

# Overview of the repository
Some description
```
enkf_landmarks
│   README.md
└───src
│   └─── enkf.py                    # main class defining the EnKF
│   └─── lddmm.py                   # defines the forward operator written in PyTorch
│   └─── example_enkf.py            # a script to generate the figures in this paper
│   └─── example_lddmm.py           # a script to demonstrate the PyTorch forward operator
│   └─── manufacture_shape_data.py  # a script generates new targets
│   └─── utils.py
└───data   # contains the targets used in the paper 
    └───TARGET_{1,2,3,A,B,C}
        └─── pe.pickle
        └─── target.pickle
        └─── target.pdf
```

# Reproducing the figures in the paper

Remember to activate the virtual environment before running the commands below.

  - The script ```manufacture_shape_data.py``` generates target shapes like the ones in figure 1. The three
  shapes in said figure as well as the initial momentum used to generate them can be found in ```data/TARGET_{1,2,3}```.
  - Figures 2-7: ```python example_enkf.py``` iterates over the 
  - ```python example_lddmm.py``` demonstrates the forward operator as well as a shooting
  approach to determine the optimal trajectories between the template and target. Generates ```example_lddmm.pdf```
  displaying these.


# License

MIT

# Todo

 - Make it work for curves or images.
 - Parallelise the EnkF.
 - Clean up numpy/torch mess
 - Use pathlib everywhere
