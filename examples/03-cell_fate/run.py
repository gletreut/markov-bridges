#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2023-08-16

@author: Guillaume Le Treut - guillaume.letreut@czbiohub.org

Simulation of Markov bridges in the Mueller-Brown potential.

execution time:
  User+Sys CPU walltime = 36:07:36.20s
  Real time on an AMD EPYC 7713 64-Core Processor = 35:46.69s
'''
#-------------------------------------------------------------------
# imports
#-------------------------------------------------------------------
import logging
logging.basicConfig(format='%(asctime)s [%(filename)s:%(lineno)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
from pathlib import Path

import numpy as np

from mbridges.markov import KMonteCarloBridge

#-------------------------------------------------------------------
# parameters
#-------------------------------------------------------------------
state_init = 1103 # initial state
categories = {'Alpha': 0, 'Beta': 1, 'Delta': 2, 'Ductal': 3, 'Epsilon': 4, 'Ngn3 high EP': 5, 'Ngn3 low EP': 6,'Pre-endocrine': 7}

nruns = 4
steps = 2**10 # maximum number of Monte Carlo steps
seed = 123 # random generator seed

xlim = [-20., 10.0]
ylim = [-6., 8.0]
macheps=1.0e-15 # smallest non-zero positive float

# bridge parameters
state_final = 3437 # final state
t_final = 100. # final time

# processing parameters
dt_regis = 0.5 # time resolution for the registration (before time rescaling by Brownian time). decrease to get more snapshots

#-------------------------------------------------------------------
# functions
#-------------------------------------------------------------------
#------ KMC ------

def run_kmc(outputdir):
  # load transition rates matrix
  fname = 'rates.dat'
  fpath = Path(Path(__file__).parent / fname)
  rates = np.loadtxt(fpath).astype('float64')
  logging.info("Transition rates matrix loaded from {}.".format(fpath))

  fname = 'coordinates.dat'
  fpath = Path(Path(__file__).parent / fname)
  coordinates = np.loadtxt(fpath).astype('float64')
  logging.info("Coordinates loaded from {}.".format(fpath))

  # runs
  trajdir = outdir / 'trajectories'
  if not trajdir.is_dir():
    trajdir.mkdir()
  ## delete existing snapshots
  for child in trajdir.iterdir(): child.unlink()

  npad = int(1+np.floor(np.log10(nruns)))
  nfmt = "{:0" + "{:d}".format(npad) + "d}"

  ## define variables for the eigenvalue decomposition of -W
  L=None
  U=None
  Uinv=None
  s = seed
  for n in range(nruns):
    logging.info("Starting run {:s}/{:s}".format(nfmt.format(n+1),nfmt.format(nruns)))
    stepper = KMonteCarloBridge(rates, state_init, state_final, t_final, seed=s, L=L, U=U, Uinv=Uinv)
    stepper.step(steps)

    logging.info("Simulation completed. Reached time T = {} in {:d} steps. mean waiting time = {:g}.".format(stepper.t, len(stepper.t_list),stepper.t/(len(stepper.t_list)-1)))

    # write trajectory
    traj = stepper.get_trajectory()
    fname = 'traj{:s}.dat'.format(nfmt.format(n+1))
    fpath = trajdir / fname
    np.savetxt(fpath, traj)

    # updates
    ## update seed
    s += 1
    ## communicate eigenvalue decomposition to next instance
    if (L is None) or (U is None) or (Uinv is None):
      L = stepper.L.copy()
      U = stepper.U.copy()
      Uinv = stepper.Uinv.copy()
  # end loop on n

#------ plots ------

def plot_trajectories_overlay(traj_list, coordinates, lw=0.5, ms=2, figsize=(4,3)):
  '''
  Plot a list of trajectories overlaid on the potential
  '''
  import opt_einsum as oe
  import matplotlib.pyplot as plt
  import matplotlib.gridspec
  import matplotlib.colors as mco
  import matplotlib.cm as cm
  from mbridges.utils.processing import register_trajectories
  from utils import plot_umap

  XYs = coordinates[:,:2]
  labels = coordinates[:,2].astype('int64')
  fig = plot_umap(XYs=XYs, labels=labels, categoriesdict=categories)
  ax = fig.gca()
  ntraj = len(traj_list)
  norm = mco.Normalize(vmin=0,vmax=ntraj-1)
  cmap = cm.rainbow

  t_regis, trajs_regis = register_trajectories(traj_list, tmin=0, tmax=t_final, dt=dt_regis)
  Xs,Ys = oe.contract('ktd->dkt', XYs[trajs_regis.astype('int64')])
  for i,(X,Y) in enumerate(zip(Xs,Ys)):
    ax.plot(X,Y,'-',lw=lw,color=cmap(norm(i)))

  # show initial and final states
  xi,yi = XYs[state_init]
  xf,yf = XYs[state_final]
  ax.plot([xi,xf], [yi,yf], 'ko', ms=4*ms, mfc='none', mew=lw)

  return fig

def plot_trajectories_snapshots(snapdir, traj_list, coordinates, lw=0.5, ms=2, figsize=(4,3), taillength=15, aspect='equal', fprefix="state_"):
  '''
  Plot a list of trajectories overlaid on the potential at distinct time points
  '''
  from mbridges.utils.processing import register_trajectories
  from utils import plot_umap, plot_trajectories


  XYs_static = coordinates[:,:2]
  labels = coordinates[:,2].astype('int64')
  func_background = lambda : plot_umap(XYs=XYs_static, labels=labels, categoriesdict=categories)

  t_regis, trajs_regis = register_trajectories(traj_list, tmin=0, tmax=t_final, dt=dt_regis)
  XYs = XYs_static[trajs_regis.astype('int64')]

  plot_trajectories(snapdir, t_regis, XYs, xlim=xlim, ylim=ylim, XYs_static=XYs_static,
      aspect=aspect, taillength=taillength, ms=ms, lw=lw, func_background=func_background, fprefix=fprefix)

  logging.info("Snapshots written to {:s}.".format(str(snapdir)))
  return

def plots(outputdir):
  logging.info("Starting plots.")
  # general parameters
  lw = 0.5
  ms = 2
  ext = '.png'
  dpi = 300
  taillength=10
  fps=30

  # ouput directory for plots
  figdir = outdir / 'figures'
  if not figdir.is_dir():
    figdir.mkdir()

  # load trajectories
  trajdir = outputdir / 'trajectories'
  file_list = sorted([f for f in trajdir.glob('traj*.dat')])
  traj_list = [np.loadtxt(f) for f in file_list]
  logging.info("Loaded {:d} trajectories.".format(len(file_list)))

  # load coordinates
  fpath = Path(Path(__file__).parent) /  'coordinates.dat'
  coordinates = np.loadtxt(fpath)
  logging.info("Loaded {:s} coordinates.".format(str(fpath)))

  # plot trajectories overlaid
  fig = plot_trajectories_overlay(traj_list, coordinates, lw=lw, ms=ms, figsize=(4,3))
  fname = 'trajectories_overlay'
  fpath = figdir / (fname + ext)
  fig.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi)
  logging.info("Written figure in {:s}.".format(str(fpath)))

  # plot snapshots in time of trajectories
  snapdir = figdir / 'snapshots'
  if not snapdir.is_dir():
    snapdir.mkdir(exist_ok=True, parents=True)
    logging.info("Created directory {:s}".format(str(snapdir)))
  ## delete existing snapshots
  for child in snapdir.glob('*.png'): child.unlink()

  plot_trajectories_snapshots(snapdir, traj_list, coordinates, lw=lw, ms=ms, figsize=(4,3), taillength=taillength, fprefix='t')

  # write movie
  from utils import make_movie
  file_list = [f for f in snapdir.glob('*.png')]
  file_list.sort()

  fpath = figdir / 'movie_trajectories.mp4'
  make_movie(fpath, file_list, fps=fps)
  logging.info("Movie written to {:s}.".format(str(fpath)))

  return


#-------------------------------------------------------------------
# script
#-------------------------------------------------------------------
if __name__ == "__main__":
  # output folder
  outdir = Path(__file__).parent / 'results'
  if not outdir.is_dir():
    outdir.mkdir()

  run_kmc(outdir)

  plots(outdir)

  logging.info("Normal exit.")
