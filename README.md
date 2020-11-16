# EnKFDiffeo
Write stuff.

# Overview of the repository
Some description
```
enkf_landmarks
│   README.md
└───src
│   └─── enkf.py                    # main class defining the EnKF
│   └─── emsemble.py                # contains a data class storing ensemble data
│   └─── lddmm.py                   # defines the forward operator written in PyTorch
│   └─── example_enkf.py            # a script to generate the figures in this paper
│   └─── example_lddmm.py           # a script to demonstrate the PyTorch forward operator
│   └─── manufacture_shape_data.py  # a script generates new targets
│   └─── utils.py
└───data  # contains targets with different 
    └───TARGET_{1,2,3,A,B,C}
        └─── pe.pickle
        └─── target.pickle
        └─── target.pdf
```

# Dependencies
```
pytorch>=1
```

# Reproducing the figures in the paper



Remember to activate the virtual environment before running the commands below.

  - Figure 1: ```$ python lddmm.py``` creates stuff.
  - Figures 2-7: ```$ python example_enkf.py``` iterates over the 

The script ```manufacture_shape_data.py``` generates new targets.


# License

MIT

# Todos

 - Make it work for curves or images.
 - Parallelise the EnkF.
    
    