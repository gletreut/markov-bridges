#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2023-08-16

@author: Guillaume Le Treut - guillaume.letreut@czbiohub.org

Simulation of Markov bridges in the Mueller-Brown potential.

execution time:
  User+Sys CPU walltime = 18:52:29.93s
  Real time on an AMD EPYC 7713 64-Core Processor = 19:18.89s
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
npts_per_axis = 2**5 + 1 # number of points along 1 dimension
xlim = [-1.5, 1.] # xmin, xmax
ylim = [-0.5, 2.] # ymin, ymax
macheps=1.0e-15 # smallest non-zero positive float
umin = -75 # lower truncation value for the potential [k_B T]
umax = 15 # upper truncation value for the potential [k_B T]

N = npts_per_axis**2 # number of states
state_init = 421 # initial state

nruns = 32
steps = 2**10 # maximum number of Monte Carlo steps
seed = 123 # random generator seed

# bridge parameters
state_final = 930 # final state
t_final = 200. # final time

# processing parameters
dt_regis = 1.0 # time resolution for the registration (before time rescaling by Brownian time). decrease to get more snapshots

#-------------------------------------------------------------------
# functions
#-------------------------------------------------------------------
#------ utility ------

def build_rate_matrix(npts_per_axis, xlim, ylim, umin, umax, macheps=1.0e-15):
  '''
  INPUT:
    npts_per_axis [int]: number of points along 1 dimension
    xlim [(float, float)]: xlim, xmax
    ylim [(float, float)]: ylim, ymax
    umin [float]: lower truncation value for the potential [k_B T]
    umax [float]: upper truncation value for the potential [k_B T]
  OUTPUT:
    rates [ndarray]: NxN matrix of transition rates
    coordinates [ndarray]: NxD matrix of coordinates and energy for the states
  '''
  from utils import potential
  import scipy.optimize


  # make the mesh
  xvals = np.linspace(xlim[0], xlim[1], npts_per_axis+2)[1:-1]
  xmesh = np.einsum('i,j->ij', xvals, np.ones(npts_per_axis))

  yvals = np.linspace(ylim[0], ylim[1], npts_per_axis+2)[1:-1]
  ymesh = np.einsum('j,i->ij', yvals, np.ones(npts_per_axis))

  # compute potential
  umesh = potential(xmesh, ymesh)

  xcoords = np.ravel(xmesh, order='C')
  ycoords = np.ravel(ymesh, order='C')
  U = np.ravel(umesh, order='C')

  # set maximum energy amplitude so that minimum equilibirum probability > epsilon
  if umin is None:
    umin = np.min(U)
  if umax is None:
    umax = np.max(U)
  logging.info("Umin = {:g}  Umax = {:g}".format(umin, umax))

  ## rought adjustment
  U[U < umin] = umin
  U[U > umax] = umax
  delta_max = -np.log(macheps)
  beta = delta_max / (umax - umin)

  ## fine adjustment
  ## find the temperature so that there is no state with Pr < macheps
  Z = lambda b: np.sum(np.exp(-b*(U-umin)))
  Z_prime = lambda b: -np.sum((U-umin)*np.exp(-b*(U-umin)))
  func = lambda b: b*(umax-umin) + np.log(Z(b)) + np.log(macheps) # condition that min(Pr) > macheps
  func_prime = lambda b: umax-umin + Z_prime(b) / Z(b)

  res = scipy.optimize.root_scalar(func, x0=beta, method='newton', fprime=func_prime)
  beta = res.root
  logging.info("beta = {:g}  T = {:g}".format(beta, 1./beta))

  U *= beta
  umin = np.min(U)
  Pr = np.exp(-(U-umin))
  Pr /= np.sum(Pr)
  logging.info("min(Pr) = {:g}".format(np.min(Pr)))

  coordinates = np.array([xcoords, ycoords, U, Pr]).T

  # compute transition rates
  ## initial calculation
  Umat = np.array([U]*len(U)).T
  rates = np.exp(-0.5*(Umat - Umat.T))

  ## set diagonal rates to 0
  np.fill_diagonal(rates, 0)

  ## only keep transitions to nearest neighbors
  ## compute the square distance matrix
  imesh = np.einsum('i,j->ij', np.arange(npts_per_axis), np.ones(npts_per_axis))
  jmesh = imesh.T
  icoords = np.ravel(imesh, order='C')
  jcoords = np.ravel(jmesh, order='C')

  X = np.concatenate([icoords.reshape(-1,1),jcoords.reshape(-1,1)], axis=1)
  U = np.ones((N,2), dtype=np.float_)
  S = X**2
  D2 = np.einsum('ia,ja->ij',S,U) + np.einsum('ia,ja->ij',U,S) - 2*np.einsum('ik,jk->ij',X,X)

  idx = (D2 <= 1.) & ~(D2 == 0.0)
  rates[~idx] = 0.

  return rates, coordinates

#------ KMC ------

def run_kmc(outputdir):
  # build transition rates matrix
  rates, coordinates = build_rate_matrix(npts_per_axis=npts_per_axis, xlim=xlim, ylim=ylim, umin=umin, umax=umax, macheps=macheps)
  fname = 'rates.dat'
  fpath = outdir / fname
  np.savetxt(fpath, rates)
  # load with np.loadtxt(fpath).astype('float64')
  logging.info("Transition rates matrix written to {}.".format(fpath))

  fname = 'coordinates.dat'
  fpath = outdir / fname
  np.savetxt(fpath, coordinates)
  # load with np.loadtxt(fpath).astype('float64')
  logging.info("Coordinates written to {}.".format(fpath))

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
  from utils import plot_potential

  XYs = coordinates[:,:2]
  Us = coordinates[:,2]
  fig = plot_potential(XYs=XYs, Us=Us, \
                       extent=[xlim[0], xlim[1], ylim[0], ylim[1]], \
                       x_minima = None, \
                       levels = np.linspace(np.min(Us), np.max(Us), 2**5+2)[1:-1], \
                       title = None)
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
  from utils import plot_potential, plot_trajectories


  XYs_static = coordinates[:,:2]
  Us = coordinates[:,2]
  func_background = lambda : plot_potential(XYs=XYs_static, Us=Us, \
                       extent=[xlim[0], xlim[1], ylim[0], ylim[1]], \
                       x_minima = None, \
                       levels = np.linspace(np.min(Us), np.max(Us), 2**5+2)[1:-1], \
                       title = None)

  t_regis, trajs_regis = register_trajectories(traj_list, tmin=0, tmax=t_final, dt=dt_regis)
  XYs = XYs_static[trajs_regis.astype('int64')]

  # rescale time by the Brownian time
  dx = (xlim[1]-xlim[0])/(npts_per_axis-1)
  dy = (ylim[1]-ylim[0])/(npts_per_axis-1)
  t_regis *= np.min([dx,dy])**2

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
  taillength=15
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
  fpath = outdir /  'coordinates.dat'
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
