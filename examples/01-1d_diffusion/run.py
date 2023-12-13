#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2023-08-16

@author: Guillaume Le Treut - guillaume.letreut@czbiohub.org

Simulation of Markov bridges following a jump process corresponding
to the biased random walk on the 1d lattice.

execution time:
  User+Sys CPU walltime = 9:13:23.87s
  Real time on an AMD EPYC 7713 64-Core Processor = 9:00.69s
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
N = 256 # number of states
alpha = 1.0 # forward rate
beta = 0.5 # backward rate
state_init = 0 # initial state

nruns = 64
steps = 2**12 # maximum number of Monte Carlo steps
seed = 123 # random generator seed

# bridge parameters
state_final = 0 # final state
t_final = 100. # final time

#-------------------------------------------------------------------
# functions
#-------------------------------------------------------------------
#------ utility ------

def build_rate_matrix(N):
  '''
  INPUT:
    N [int]: size of the matrix
  OUTPUT:
    rates [ndarray]: NxN matrix of transition rates
  '''

  rates = np.zeros((N,N), dtype=np.float_)
  for i in range(1,N-1):
    rates[i,i+1] = beta
    rates[i,i-1] = alpha

  # periodic boundary conditions
  rates[0,1] = beta
  rates[0,N-1] = alpha
  rates[N-1,0] = beta
  rates[N-1,N-2] = alpha

  return rates

class DiffusionProcess1D:
  def __init__(self, N, alpha, beta, n_init = 0, n_final = 0):
    self.N = N
    self.alpha = alpha
    self.beta = beta
    self.P_init = np.float_(np.arange(N) == n_init)
    self.Ptilde_init = np.fft.fft(self.P_init)
    self.Q_final = np.float_(np.arange(N) == n_final)
    self.Qtilde_final = np.fft.fft(self.Q_final)

    self.omega = np.zeros(self.N, dtype=np.float_)
    self.compute_omega()
    self.omega_tilde = np.fft.fft(self.omega)

  def compute_omega(self):
    self.omega = np.zeros(self.N, dtype=np.float_)
    self.omega[0] = self.alpha + self.beta
    self.omega[1] = -self.alpha
    self.omega[-1] = -self.beta
    return

  def get_prob_forward(self, delta_t):
    """
    compute the forward probability of the diffusion process
    """
    Ptilde = np.exp(-self.omega_tilde*delta_t)*self.Ptilde_init
    P = np.real(np.fft.ifft(Ptilde))
    return P, Ptilde

  def get_prob_backward(self, delta_t):
    """
    compute the backward probability of the diffusion process
    """
    # Qtilde = np.exp(np.transpose(self.omega_tilde)*delta_t)*self.Qtilde_final
    Qtilde = np.exp(-np.conjugate(self.omega_tilde)*delta_t)*self.Qtilde_final
    Q = np.real(np.fft.ifft(Qtilde))
    return Q, Qtilde

#------ KMC ------

def run_kmc(outputdir):
  # build transition rates matrix
  rates = build_rate_matrix(N)
  fname = 'rates.dat'
  fpath = outdir / fname
  np.savetxt(fpath, rates)
  # load with np.loadtxt(fpath).astype('float64')
  logging.info("Transition rates matrix written to {}.".format(fpath))

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

def plot_trajectories(traj_list, lw=0.5, ms=2, figsize=(8,3), wraporigin=False, thetas_grid = None, times_grid = None):
  '''
  Plot a list of trajectories
  '''
  import matplotlib.pyplot as plt
  import matplotlib.gridspec
  import matplotlib.colors as mco
  import matplotlib.cm as cm

  ntraj = len(traj_list)
  norm = mco.Normalize(vmin=0,vmax=ntraj-1)
  cmap = cm.rainbow

  fig = plt.figure(facecolor='w', figsize=figsize)
  ax1 = fig.add_subplot(1,2,1)
  ax2 = fig.add_subplot(1,2,2, projection='polar')

  for i in range(ntraj):
    T,X = traj_list[i].T
    color = cmap(norm(i))

    # polar plot
    ax2.plot(X*2.*np.pi/N,T, ls='-', lw=lw, marker='o', ms=ms, color=color)

    # time series plot
    ## wrap around origin
    if wraporigin:
      idx = X > N//2
      X[idx] -= N

    ## make jumps apparent
    Tpc = np.array([T[:-1],T[1:]]).T.ravel(order='C')
    Xpc = np.array([X[:-1],X[:-1]]).T.ravel(order='C')
    Tpc = np.append(Tpc, T[-1])
    Xpc = np.append(Xpc, X[-1])

    ax1.plot(Tpc, Xpc, ls='-', lw=lw, marker='o', ms=ms, color=color)

  ax1.set_xlabel('t', fontsize='medium')
  ax1.set_ylabel('i', fontsize='medium')
  yticks = ax1.get_yticks()
  if wraporigin:
    yticklabels = []
    for y in yticks:
      if y < 0:
        y += N
      yticklabels.append("{:.0f}".format(y))
    ax1.set_yticklabels(yticklabels)
  ax1.tick_params(axis='both', labelbottom=True, labeltop=False, labelright=False, labelleft=False)

  if thetas_grid is None:
    val = 2*np.pi * (np.arange(N)+1) / N *180./np.pi
    label = ["{:d}".format(n) for n in range(N)]
    npts = len(val)
    dn = max(1, npts // 8) # default to 8 angle values
    thetas_grid = [val[::dn], label[::dn]]
  ax2.set_thetagrids(thetas_grid[0], labels = thetas_grid[1], fontsize='medium')

  if times_grid is None:
    tmin = 0.
    tmax = np.max([np.max(traj[:,0]) for traj in traj_list])
    val = np.linspace(tmin, tmax, 8+1)[1:] # default to 8 concentric lines
    label = ["{:.1f}".format(t) for t in val]
    times_grid = [val, label]

  ax2.set_rgrids(times_grid[0], labels=times_grid[1], angle=90, fontsize='medium')
  ax2.set_rmin(0.)
  ax2.set_rmax(times_grid[0][-1])

  return fig

def plot_proba(traj_list, bin_sel, delta_t, lw=0.5, ms=2, figsize=(4,3)):
  '''
  Plot a the probability distribution for several times (through `bin_sel`)
  '''
  import matplotlib.pyplot as plt
  import matplotlib.gridspec
  import matplotlib.colors as mco
  import matplotlib.cm as cm

  # compute time bins
  T_all, X_all = np.concatenate(traj_list, axis=0).T[:2]
  a = 0
  b = np.ceil(np.max(T_all))
  nbins = int((b-a)/delta_t)
  delta_t = (b-a)/nbins
  t_edges = a + np.arange(nbins+1)*delta_t
  digitized = np.digitize(T_all, t_edges)

  # compute probabilities from trajectories
  X = np.arange(N)
  Ps = []
  errs = []
  for k in range(len(bin_sel)):
    bin_val = bin_sel[k]
    idx = digitized == (bin_val+1)
    data = X_all[idx]
    P = np.array([np.sum(data == x)/ len(data) for x in X])
    err = np.sqrt(P*(1-P)/len(data))
    Ps.append(P)
    errs.append(err)

  # define a Diffusion1D process to compute the theoretical R(t) and P(t)
  process = DiffusionProcess1D(N, alpha, beta, n_init=state_init, n_final=state_final)

  # make figure
  fig = plt.figure(facecolor='w', figsize=figsize)
  ax = fig.gca()
  norm = mco.Normalize(vmin=0,vmax=len(bin_sel)-1)
  cmap = cm.rainbow

  for k in range(len(bin_sel)):
    bin_val = bin_sel[k]
    tval = 0.5*(t_edges[:-1]+t_edges[1:])[bin_val]
    color = cmap(norm(k))
    label = 't = {:g}'.format(tval)
    P = Ps[k]
    err = errs[k]
    idx = err > 0.

    ax.errorbar(X[idx], P[idx], yerr=err[idx], marker='.', ms=ms, elinewidth=lw, ls='none', lw=lw, capsize=ms, color=color, label=label)

    P,Ptilde = process.get_prob_forward(delta_t = tval)
    Q,Qtilde = process.get_prob_backward(delta_t = (t_final-tval))
    R = P*Q
    rnorm = np.sum(R)
    R /= rnorm
    ax.plot(R, '-', lw=lw, color=color)
    ax.plot(P, '--', lw=lw, color=color)

  ax.set_xlim(0,20)
  ax.set_ylim(0,0.3)
  ax.set_xlabel('x', fontsize='medium')
  ax.set_ylabel('pdf', fontsize='medium')

  ax.legend(loc='best', fontsize='medium', frameon=False)

  return fig

def plots(outputdir):
  logging.info("Starting plots.")
  # general parameters
  lw = 0.5
  ms = 2
  ext = '.png'
  dpi = 300

  # ouput directory for plots
  figdir = outdir / 'figures'
  if not figdir.is_dir():
    figdir.mkdir()

  # load trajectories
  trajdir = outputdir / 'trajectories'
  file_list = sorted([f for f in trajdir.glob('traj*.dat')])
  traj_list = [np.loadtxt(f) for f in file_list]
  logging.info("Loaded {:d} trajectories.".format(len(file_list)))

  # plot trajectories as time series
  ntraj = 4
  times_grid = list(zip(*[[t,"{:.0f}".format(t)] for t in [20, 40, 60, 80, 100]]))
  fig = plot_trajectories([traj[:,:2] for traj in traj_list[:ntraj]], \
                          times_grid = times_grid,\
                          wraporigin=True, \
                          lw=lw, ms=ms)
  fname = 'trajectories'
  fpath = figdir / (fname + ext)
  fig.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi)
  logging.info("Written figure in {:s}.".format(str(fpath)))

  # plot the master equation probability
  bin_sel = [1, 5, 10, 50]
  delta_t = 1.

  fig = plot_proba(traj_list, bin_sel=bin_sel, delta_t=delta_t, lw=lw, ms=ms)

  fname = 'probabilities'
  fpath = figdir / (fname + ext)
  fig.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi)
  logging.info("Written figure in {:s}.".format(str(fpath)))

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
