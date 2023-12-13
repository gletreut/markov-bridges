# Markov bridges package

This repository provides an implementation as a Python package to sample *Markov bridges* as described in the article [arXiv:2312.08605](https://arxiv.org/abs/2312.08605).

The `mbridges` package can be installed with:
```
python3 -m pip install .
```

Below we provide some instructions to run the examples provided in `examples/`.

## Setup

We recommend using [pyenv](https://github.com/pyenv/pyenv) for managing
different versions of Python, and
[venv](https://docs.python.org/3/library/venv.html) for creating virtual
environments.

Given the provided `.python-version` file, the following command should return
`Python 3.12.1`:
```
python -V
```

A local virtual environment can be initialized with:
```
python -m venv .venv
source .venv/bin/activate
```

The packages required to run the examples can be installed using `pip`:
```
python -m pip install -r requirements.txt
```

## Example 1: jump process for the biased random walk on the 1d lattice

Run:
```
python examples/01-1d_diffusion/run.py
```

The results of this program will be written under
`examples/01-1d_diffusion/results` as shown below.
```
.
├── results
│   ├── figures
│   │   ├── probabilities.png
│   │   └── trajectories.png
│   ├── rates.dat
│   └── trajectories
│       ├── traj01.dat
│       ├── traj02.dat
│       ·
│       ·
│       ·
│       └── traj64.dat
└── run.py

3 directories, 68 files
```

Each `trajXX.dat` file contains one Kinetic Monte Carlo trajectory (KMC), with
the first column corresponding to the time t of a jump and the second column
corresponding to the state reached at the corresponding jump.

The figure `trajectories.png` shows a few trajectories represented as time
series, and the figure `probabilities.png` show the probability distribution
estimated from the trajectories for a few time points.

## Example 2: Markov bridges in the Müller-Brown potential

Run:
```
python examples/02-mueller_brown/run.py
```

The results of this program will be written under `examples/02-mueller_brown/results` as shown below:
```
.
├── results
│   ├── coordinates.dat
│   ├── figures
│   │   ├── movie_trajectories.mp4
│   │   ├── snapshots
│   │   │   ├── t000.png
│   │   │   ·
│   │   │   ·
│   │   │   ·
│   │   │   └── t200.png
│   │   └── trajectories_overlay.png
│   ├── rates.dat
│   └── trajectories
│       ├── traj01.dat
│       ·
│       ·
│       ·
│       └── traj32.dat
├── run.py
└── utils.py

4 directories, 239 files
```

The figure `trajectories_overlay.png` shows an overlay of all trajectories for
all times, and the snapshots `tXXX.png` show the state of all trajectories
at a given time. A video of the snapshots can be found in
`movie_trajectories.mp4`.

## Example 3: Markov bridges to analyze cell-fate choices

For this example, we provide input transition rates in `rates.dat` and the UMAP
coordinates for the corresponding states in `coordinates.dat` (see publication
for details).

Run:
```
python examples/03-cell_fate/run.py
```

The results of this program will be written under `examples/03-cell_fate/results` as shown below:
```
.
├── coordinates.dat
├── rates.dat
├── results
│   ├── figures
│   │   ├── movie_trajectories.mp4
│   │   ├── snapshots
│   │   │   ├── t000.png
│   │   │   ·
│   │   │   ·
│   │   │   ·
│   │   │   └── t200.png
│   │   └── trajectories_overlay.png
│   └── trajectories
│       ├── traj1.dat
│       ├── traj2.dat
│       ├── traj3.dat
│       └── traj4.dat
├── run.py
└── utils.py

4 directories, 211 files
```

The figure `trajectories_overlay.png` shows an overlay of all trajectories for
all times, and the snapshots `tXXX.png` show the state of all trajectories
at a given time. A video of the snapshots can be found in
`movie_trajectories.mp4`.

## GPU acceleration
To use GPU acceleration for linear algebra operations such as matrix-vector
products, set the argument `linalg_module='cupy'` when initializing an instance of `KMonteCarloBridge`. This requires a working installation of the Python package [Cupy](https://cupy.dev/).
