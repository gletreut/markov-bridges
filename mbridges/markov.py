#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Dec 2022

@author: Guillaume Le Treut

"""
#==============================================================================
# libraries
#==============================================================================
import logging
import numpy as np
import opt_einsum as oe
from mbridges.utils.cupy import get_array_module

logging.basicConfig(format='%(asctime)s [%(filename)s:%(lineno)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
# logging.basicConfig(format='%(asctime)s [%(filename)s:%(lineno)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

#==============================================================================
# master equation
#==============================================================================
def rates_to_W_matrix(rates):
  '''
  Construct a W-matrix from the matrix of transition rates
  '''
  W = np.copy(rates)
  np.fill_diagonal(W, np.diagonal(W) - oe.contract('ba->a', W))
  return W

def get_sources(rates):
  '''
  Compute the states which are sources: prob(leave) = 1.
  '''
  return np.argwhere(~((oe.contract('ij->i', rates)-np.diagonal(rates)) > 0.))[:,0]

def get_absorbers(rates):
  '''
  Compute the states which are absorbing: prob(stay) = 1.
  '''
  return np.argwhere(~((oe.contract('ji->i', rates)-np.diagonal(rates)) > 0.))[:,0]

def get_Q(afinal, tleft, L, U, Uinv, ntrunc=None, eps=1.0e-12, verbose=True):
  '''
  INPUT:
    * afinal [integer]: final state
    * a [integer]: current state
    * tleft [double]: t_f-t is the time left to reach the final state
    * L [Nx1]: vector of eigenvalues of -(W-matrix) associated to the rates. They are all stricly positive but one equal to zero. Sorted by increasing value so that L[0] = 0, and L[i] > 0 for i > 0.
    * U [NxN]: Mat_Id(Bcan, Beigen) such that U_ai is the coordinate a in the canonical basis of eigenvector U^(i)
    * Uinv [NxN]: inverse of U.
    * ntrunc [integer]: sum of `ntrunc` first eigenvalues.

    Note: the W-matrix is equal to W = -U Diag(L) Uinv (einsum('ia,a,aj->ij',U,-L,Uinv))

  OUTPUT:
    * Q [Nx1]: backward probabilities
  '''
  xp = get_array_module(U)

  if (ntrunc is None) and (tleft > 0.):
    idx = ~(-L/np.log(eps) > 1./tleft)
    if not np.all(idx):
      ntrunc = np.sum(idx)

  Q = xp.real(oe.contract('i,ia,i->a', U[afinal][:ntrunc], Uinv[:ntrunc], xp.exp(-tleft*L[:ntrunc])))

  if xp.__name__ == 'cupy': # return a numpy array
    Q = Q.get()


  # Q must be non-negative
  idx = (Q > eps)
  Q[~idx] = 0.0 # to prevent any jump to those states

  # verbose
  if verbose:
    nz = np.sum(np.int_(~idx))
    if (nz > 0):
      print("#{{Q = 0}} = {:d}".format(nz))

  return Q

def get_P(ainit, t, L, U, Uinv, ntrunc=None, eps=1.0e-12, verbose=True):
  '''
  INPUT:
    * ainit [integer]: initial state
    * a [integer]: current state
    * t [double]: t is the time elapsed since the initial condition
    * L [Nx1]: vector of eigenvalues of -(W-matrix) associated to the rates. They are all stricly positive but one equal to zero. Sorted by increasing value so that L[0] = 0, and L[i] > 0 for i > 0.
    * U [NxN]: Mat_Id(Bcan, Beigen) such that U_ai is the coordinate a in the canonical basis of eigenvector U^(i)
    * Uinv [NxN]: inverse of U.
    * ntrunc [integer]: sum of `ntrunc` first eigenvalues.

    Note: the W-matrix is equal to W = -U Diag(L) Uinv (einsum('ia,a,aj->ij',U,-L,Uinv))

  OUTPUT:
    * P [Nx1]: forward probabilities
  '''
  xp = get_array_module(U)

  if (ntrunc is None) and (t > 0.):
    idx = ~(-L/np.log(eps) > 1./t)
    if not np.all(idx):
      ntrunc = np.sum(idx)

  P = xp.real(oe.contract('ai,i,i->a', U[:,:ntrunc], Uinv[:ntrunc, ainit], xp.exp(-t*L[:ntrunc])))

  if xp.__name__ == 'cupy': # return a numpy array
    P = P.get()


  # Q must be non-negative
  idx = (P > eps)
  P[~idx] = 0.0 # to prevent any jump to those states

  # verbose
  if verbose:
    nz = np.sum(np.int_(~idx))
    if (nz > 0):
      print("#{{P = 0}} = {:d}".format(nz))

  return P

def get_R(ainit, afinal, t, tleft, L, U, Uinv, ntrunc=None, eps=1.0e-12, verbose=True):
  '''
  INPUT:
    * ainit [integer]: initial state
    * afinal [integer]: final state
    * t [double]: t is the time elapsed since the initial condition
    * tleft [double]: t_f-t is the time left to reach the final state
    * L [Nx1]: vector of eigenvalues of -(W-matrix) associated to the rates. They are all stricly positive but one equal to zero. Sorted by increasing value so that L[0] = 0, and L[i] > 0 for i > 0.
    * U [NxN]: Mat_Id(Bcan, Beigen) such that U_ai is the coordinate a in the canonical basis of eigenvector U^(i)
    * Uinv [NxN]: inverse of U.
    * ntrunc [integer]: sum of `ntrunc` first eigenvalues.

    Note: the W-matrix is equal to W = -U Diag(L) Uinv (einsum('ia,a,aj->ij',U,-L,Uinv))

  OUTPUT:
    * P [Nx1]: forward probabilities
  '''
  P = get_P(ainit, t, L, U, Uinv, ntrunc, eps, verbose)
  Q = get_Q(afinal, tleft, L, U, Uinv, ntrunc, eps, verbose)

  R = P*Q
  R /= np.sum(R)

  return R

#==============================================================================
# Gillespie algorithm
#==============================================================================
def get_dwell_time(erate, u, tau_max):
  '''
  Draw a random dwell time
  INPUT:
    * erate [lambda tau: erate(tau)]: function returning the escape rate at time t+tau (time-dependent, t is implicitly managed by the user).
    * u [int]: number drawn in the interval [0,1]
  OUTPUT:
    * return the dwell duration from time 0
  '''

  from scipy.integrate import quad
  from scipy.optimize import root_scalar
  import warnings


  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    rfunc_withargs = lambda tau, **kwargs: quad(erate, 0, tau, **kwargs)[0] + np.log(1.-u)
    rfunc = lambda tau: rfunc_withargs(tau, epsrel=1.0e-8, limit=50) # TODO: this is time consuming and inefficient

    # normal root finding
    res = lambda : None # dummy lambda function to have a variable with attributes in case of error
    logging.debug("    root_scalar<newton>")
    try:
      res = root_scalar(rfunc, x0=0., method='newton', fprime=erate)
    except ValueError as e:
      res.converged=False

    if res.converged:
      return res.root

    # if not converged, we might be close to the end
    if not res.converged:
      stop = False
      atol = 1.0e-8
      npts = 10
      x0 = 0.
      x1 = 0.
      while (not stop):
        if (tau_max - x1) < atol:
          logging.info("tau_max - tau is smaller than {:g}! Returning tau_max = {:g}.".format(atol, tau_max))
          return tau_max

        delta_t = (tau_max-x1) / npts
        logging.debug("  npts = {:d},  tau_max = {:g},  x1 = {:g},  tau_max-x1 = {:g},  delta_t = {:g}".format(npts, tau_max, x1, tau_max-x1, delta_t))
        for k in range(1, npts):
          x1 += delta_t
          f1 = rfunc(x1)
          logging.debug("    x1 = {:g}    f1 = {:g}".format(x1, f1))
          if f1 > 0.:
            stop = True
            x0 = x1 - delta_t
            break
        # end for loop
      # end while loop

      logging.debug("Found x1 = {:g}".format(x1))
      logging.debug("    root_scalar<brentq>")
      res = root_scalar(rfunc, bracket=[x0, x1], method='brentq')

      if not res.converged:
        raise RuntimeError("Brentq method failed!")
      else:
        return res.root

def get_new_state(trates, u):
  '''
  Draw a new state given some rates.
  INPUT:
    * rates [Nx1]: transition rates from current state
    * u [int]: number drawn in the interval [0,1]
  OUTPUT:
    * inew [int]: new rate
  '''
  cml = np.cumsum(trates)
  cml /= cml[-1] # normalize in [0,1]

  return np.searchsorted(cml, u, side='left')

class KMonteCarlo:
  def __init__(self, rates, i_init, seed=123):
    self.i = i_init
    self.t = 0
    self.rates = rates
    self.i_list = [self.i]
    self.t_list = [self.t]

    np.random.seed(seed)

    # np.seterr(all='warn')

    # checks
    if np.any(self.rates.diagonal() != 0.):
      raise ValueError("Diagonal rates must be zero!")

    logging.info("KMonteCarlo object initialized.")

  def step(self, itmax=1):
    for it in range(itmax):
      trates = self.rates[:,self.i]

      #""" KMC
      # compute escape rate
      erate = np.sum(trates)

      # get dwell time
      u = np.random.rand()
      tau = -np.log(1.-u)/erate
      #"""

      # draw a new state
      u = np.random.rand()
      inew = get_new_state(trates, u)

      # updates
      self.i = inew
      self.t += tau
      self.i_list.append(self.i)
      self.t_list.append(self.t)

    return True

  def get_trajectory(self):
    '''
    Return trajectory as a numpy array
    '''
    return np.array([self.t_list, self.i_list]).T

  def clear(self):
    '''
    clear list variables
    '''
    self.i_list.clear()
    self.t_list.clear()
    return

class KMonteCarloBridge(KMonteCarlo):
  def __init__(self, rates, i_init, i_final, t_final, seed=123, macheps=1.0e-15, Peq=None, L=None, U=None, Uinv=None, linalg_module='numpy'):
    # call parent constructor
    super().__init__(rates, i_init, seed)

    self.macheps = macheps

    # custom construction
    ## save final state
    self.i_final = i_final
    self.t_final = t_final
    self.q_list = []

    # check the rate matrix
    sources = get_sources(rates)
    if len(sources) > 0:
      logging.info("Non irreducible Markov chain. The following states are (absolute) sources: " + ", ".join(["{:d}".format(i) for i in sources]))
      # raise RuntimeError
    absorbers = get_absorbers(rates)
    if len(absorbers) > 0:
      logging.info("Non irreducible Markov chain. The following states are (absolute) absorbers: " + ", ".join(["{:d}".format(i) for i in absorbers]))
      # raise RuntimeError

    self.W = rates_to_W_matrix(rates.astype('float64'))
    if linalg_module == 'cupy':
      try:
        import cupy as xp
      except ImportError:
        import numpy as xp
    elif linalg_module == 'numpy':
      import numpy as xp
    else:
      raise ValueError("Unknown module requested: {:s}".format(linalg_module))
    logging.info("Module used for linear algebra: {:s}.".format(xp.__name__))

    ## diagonalize -W
    if not ((L is None) or (U is None) or (Uinv is None)):
      logging.info("Loading precomputed eingenvalue decomposition.")
      self.L = L
      self.U = U
      self.Uinv = Uinv
      self.W = xp.asarray(self.W)
      idx = np.argsort(self.L.real)
      self.L = xp.asarray(self.L[idx])
      self.U = xp.asarray(self.U[:, idx])
      self.Uinv = xp.asarray(self.Uinv[idx, :])
    else:
      if Peq is None: # attempt a direct diagonalization
        logging.info("Computing eingenvalue decomposition directly.")
        self.L, self.U = np.linalg.eig(-self.W)  # because cupy only implements eigh
        idx = np.argsort(self.L.real)
        self.W = xp.asarray(self.W)
        self.L = xp.asarray(self.L[idx])
        self.U = xp.asarray(self.U[:, idx])
        self.Uinv = xp.asarray(xp.linalg.inv(self.U))
      else:
        logging.info("Computing eingenvalue decomposition using supplied values of Peq.")
        if len(Peq) != rates.shape[0]:
          raise ValueError("Peq must be of length {:d}".format(rates.shape[0]))

        sPeq = xp.sqrt(xp.asarray(Peq))
        V = oe.contract('i,ij,j->ij', 1./sPeq, self.W, sPeq) # this matrix is symmetric
        self.L, self.U = xp.linalg.eigh(-V)
        idx = xp.argsort(self.L)
        self.L = self.L[idx]
        U = U[:, idx]
        self.U = oe.contract('ia,i->ia', U, sPeq)
        self.Uinv = oe.contract('ia,i->ai', U, 1./sPeq)

      logging.info("setting smallest eigenvalue ({:g}) to 0.".format(self.L[0]))
      self.L[0] = 0.

    dist = float(xp.linalg.norm(oe.contract('ia,a,aj->ij', self.U, self.L,self.Uinv) + self.W))
    logging.info("||U L U* + W|| = {:g}".format(dist))
    err = float(dist/xp.linalg.norm(self.W))
    logging.info("||U L U* + W|| / ||W|| = {:g}".format(err))

    # compute Q
    self.Q = get_Q(
      self.i_final, self.t_final - self.t, self.L, self.U, self.Uinv, verbose=False, eps=self.macheps)
    # self.Q = get_Q_expm(self.i_final, self.t_final - self.t, self.W, verbose=False, eps=self.macheps)
    self.q_list.append(self.Q[self.i])

    logging.info("KMonteCarloBridge object initialized.")

  def get_trates(self, t):
    '''
    convenience function to return the transition rates at time t
    '''
    Q = get_Q( self.i_final, self.t_final - t, self.L, self.U, self.Uinv, verbose=False, eps=self.macheps)

    import warnings

    if not (t < self.t_final):
      raise ValueError("t = {:g} should be smaller than t_f = {:g}".format(t, self.t_final))

    with warnings.catch_warnings():
      warnings.simplefilter('error')
      try:
        trates = self.rates[:,self.i]*Q / Q[self.i]
      except Warning:
        logging.debug("    i = {:d}  t = {:g}  q_i = {:g}".format(self.i, t, Q[self.i]))
        raise RuntimeError("Problem in computation of trates! q_i is most likely zero!")

    return trates

  def check_errors(self, eps=1.0e-12):
    '''
    run some tests and raise error if necessary
    '''

    if (self.Q[self.i] < eps):
      raise ValueError("Pr(t_f i_f | t i) = 0! i = {:d}  t = {:g}  q_i = {:g}".format(self.i, self.t, self.Q[self.i]))

    return

  def step(self, itmax=1):
    for it in range(itmax):
      logging.debug("it = {:d} i = {:d}  t = {:g}  q_i = {:g}".format(it, self.i, self.t, self.Q[self.i]))
      # compute escape rate
      erate_func = lambda tau: np.sum(self.get_trates(self.t + tau))

      # get dwell time
      u = np.random.rand()
      logging.debug("  u_1 = {:g}".format(u))
      tau_max = self.t_final - self.t
      tau = get_dwell_time(erate_func, u, tau_max)

      # stop condition
      if not (tau < tau_max):
        logging.info("Next jump time is longer than t_f - t.")
        if self.i == self.i_final:
          logging.info("Normal exit.")
        else:
          logging.warning("i = {:d} <> i_f = {:d}! Something is wrong!.".format(self.i, self.i_final))
        return True

      logging.debug("  tau = {:g}".format(tau))
      # draw a new state
      u = np.random.rand()
      inew = get_new_state(self.get_trates(self.t + tau), u)
      logging.debug("  u_2 = {:g}".format(u))
      logging.debug("  tnew = {:g}  inew = {:d}".format(self.t+tau, inew))

      # updates
      self.i = inew
      self.t += tau
      self.Q = get_Q( self.i_final, self.t_final - self.t, self.L, self.U, self.Uinv, verbose=False, eps=self.macheps)
      self.i_list.append(self.i)
      self.t_list.append(self.t)
      self.q_list.append(self.Q[self.i])

      # checks
      # # errors
      self.check_errors(eps=self.macheps)

    return False

  def get_trajectory(self):
    '''
    Return trajectory as a numpy array
    '''
    traj = super().get_trajectory()
    return np.concatenate([traj, np.array(self.q_list).reshape(-1,1)], axis=1)

  def clear(self):
    '''
    clear list variables
    '''
    super().clear()
    self.q_list.clear()
    return

